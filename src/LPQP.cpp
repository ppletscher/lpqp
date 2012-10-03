#include <LPQP.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>

LPQP::LPQP(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges):
    _theta_unary(theta_unary),
    _theta_pair(theta_pair),
    _edges(edges),
    _neighbors(),
    _rho_start(1e-1),
    _rho_end(1e10),
    _eps_entropy(1e-3),
    _eps_dkl(1e-6),
    _eps_obj(1e-3),
    _eps_mp(1e-4),
    _num_max_iter_dc(60),
    _num_max_iter_mp(300),
    _rho_schedule_constant(1.5),
    _initial_lp_active(false),
    _initial_lp_improvement_ratio(1e-2),
    _initial_lp_rho_start(25),
    _initial_rho_similar_values(false),
    _initial_rho_factor_kl_smaller(10),
    _skip_if_increase(true),
    _num_vertices(theta_unary.size()),
    _num_edges(theta_pair.size()),
    _theta_unary_curr(theta_unary),
    _mu_unary(),
    _mu_pair(),
    _history_obj(),
    _history_obj_qp(),
    _history_obj_lp(),
    _history_obj_decoded(),
    _history_iter(),
    _history_rho()
{
    // input checks
    assert(_theta_pair.size() == _num_edges);
    assert(_edges.cols() == static_cast<int>(_num_edges));
    
    // number of states for each variable
    _num_states.resize(_num_vertices);
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        _num_states[v_idx] = _theta_unary[v_idx].size();
    }
    
    // initialize unary marginals
    _mu_unary.resize(_num_vertices);
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        _mu_unary[v_idx].setConstant(_num_states[v_idx], 1.0/static_cast<double>(_num_states[v_idx]));
    }
    
    // initialize pairwise marginals and messages (duals of marginals)
    _mu_pair.resize(_num_edges);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);
    
        _mu_pair[e_idx].setConstant(_num_states[i]*_num_states[j], 1.0/static_cast<double>(_num_states[i]*_num_states[j]));
    }

    // initialize the graph data structure from the edges
    _neighbors.resize(_num_vertices);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _neighbors[i].push_back(std::make_pair(j, e_idx));
        _neighbors[j].push_back(std::make_pair(i, e_idx));
    }
}

LPQP::~LPQP()
{

}

std::vector< Eigen::VectorXd >& LPQP::getUnaryMarginals()
{
    return _mu_unary;
}

std::vector< Eigen::VectorXd >& LPQP::getPairwiseMarginals()
{
    return _mu_pair;
}

double LPQP::computeObjectiveDecoded()
{
    double energy = 0.0;
    
    // unaries
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        size_t x;
        _mu_unary[v_idx].maxCoeff(&x);
        energy += _theta_unary[v_idx][x];
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        
        size_t xi, xj;
        _mu_unary[i].maxCoeff(&xi);
        _mu_unary[j].maxCoeff(&xj);
        
        energy += _theta_pair[e_idx](xi+_num_states[i]*xj);
    }

    return energy;
}

Eigen::VectorXi LPQP::getDecodedAssignment()
{
    Eigen::VectorXi decoded;
    decoded.resize(_num_vertices);
    
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        size_t x;
        _mu_unary[v_idx].maxCoeff(&x);
        decoded[v_idx] = x;
    }

    return decoded;
}

double LPQP::computeQPValue()
{
    double energy = 0.0;

    // unaries
    for (size_t i=0; i<_num_vertices; i++) {
        energy += _theta_unary[i].transpose()*_mu_unary[i];
        for (size_t k=0; k<_num_states[i]; k++) {
            assert(_mu_unary[i][k]>=0);
        }
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        for (size_t ki=0; ki<_num_states[i]; ki++) {
            for (size_t kj=0; kj<_num_states[j]; kj++) {
                energy += _theta_pair[e_idx](ki+_num_states[i]*kj)*_mu_unary[i](ki)*_mu_unary[j](kj);
            }
        }
    }

    return energy;
}

double LPQP::computeLPValue()
{
    double energy = 0.0;

    // unaries
    for (size_t i=0; i<_num_vertices; i++) {
        energy += _theta_unary[i].transpose()*_mu_unary[i];
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        energy += _theta_pair[e_idx].transpose()*_mu_pair[e_idx];
    }

    return energy;
}

double LPQP::computeObjective(double rho)
{
    return computeLPValue() + rho*computeKLDivergence();
}

double LPQP::computeUnaryEntropy()
{
    double entropy = 0.0;

    for (size_t i=0; i<_num_vertices; i++) {
        for (size_t k=0; k<_num_states[i]; k++) {
            assert(_mu_unary[i](k) >= 0);
            entropy -= _mu_unary[i](k)*log0(_mu_unary[i](k));
        }
    }

    return entropy;
}

void LPQP::printProgressHeader()
{
    printf("%10s | %7s | %7s | %10s | %10s | %10s | %10s\n", "rho", "DC iter", "MP iter", "obj", "QP obj", "LP obj", "dec obj");
}

void LPQP::printProgress(double rho, size_t dc_iter, size_t mp_iter, double val, double val_qp, double val_lp, double val_decoded)
{
    printf("%10.6g   %7d   %7d   %10.6g   %10.6g   %10.6g   %10.6g\n", rho, \
            static_cast<int>(dc_iter), static_cast<int>(mp_iter), val, val_qp,
            val_lp, val_decoded);
}

void LPQP::setRhoStart(double rho_start)
{
    _rho_start = rho_start;
}

void LPQP::setRhoEnd(double rho_end)
{
    _rho_end = rho_end;
}

void LPQP::setRhoScheduleConstant(double c)
{
    _rho_schedule_constant = c;
}

void LPQP::setMaximumNumberOfIterationsDC(size_t num_max_iter)
{
    _num_max_iter_dc = num_max_iter;
}

void LPQP::roundSolution()
{
    // based on the QP rounding scheme by Ravikumar and Lafferty in:
    // "Quadratic Programming Relaxations for Metric Labeling and Markov
    // Random Field MAP Estimation", ICML 2006.

    // For every node establish the assignment by max_(k) of : theta_i,k + sum_(j neighbor of i,j) theta_ik;jl
    for (size_t i=0; i<_num_vertices; i++) {

        int y_i = -1; // no assignment yet
        double val = 0;

        // go over all states of node i
        for (size_t ki=0; ki<_num_states[i]; ki++) {

            double sum_over_neighbors = 0;
            for (size_t n_idx=0; n_idx< _neighbors[i].size(); n_idx++) {

                size_t j = _neighbors[i][n_idx].first;
                size_t e_idx = _neighbors[i][n_idx].second;

                for (size_t kj=0; kj<_num_states[j]; kj++) {
                    if (i == static_cast<size_t>(_edges(0, e_idx))) {
                        sum_over_neighbors -= _theta_pair[e_idx](ki+_num_states[i]*kj)*_mu_unary[j](kj);
                    }
                    else {
                        sum_over_neighbors -= _theta_pair[e_idx](kj+_num_states[j]*ki)*_mu_unary[j](kj);
                    }
                }
            }

            sum_over_neighbors -= _theta_unary[i][ki];

            if(y_i == -1){
                val = sum_over_neighbors;
                y_i = 0;
            }
            else if(sum_over_neighbors > val){
                y_i = ki;
                val = sum_over_neighbors;
            }
        }

        assert(y_i >= 0);

        // set the mu unaries according to the assignment y_i
        _mu_unary[i].setZero(_num_states[i]);
        _mu_unary[i](y_i) = 1.0;
    }

    // now set the mu pairs accordingly
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {

        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);

        size_t xi, xj;
        _mu_unary[i].maxCoeff(&xi);
        _mu_unary[j].maxCoeff(&xj);

        _mu_pair[e_idx].setZero(_num_states[i]*_num_states[j]);
        _mu_pair[e_idx](xi+_num_states[i]*xj) = 1.0;
    }
}

double LPQP::determineInitialRho()
{
    double obj_lp = std::abs(computeLPValue());
    double obj_kl = computeKLDivergence();

    if (obj_kl <= 0) {
        return _rho_start;
    }
    else {
        return (obj_lp/(obj_kl))/_initial_rho_factor_kl_smaller;
    }
}

void LPQP::setEpsilonEntropy(double eps_entropy)
{
    _eps_entropy = eps_entropy;
}

void LPQP::setEpsilonKullbackLeibler(double eps_dkl)
{
    _eps_dkl = eps_dkl;
}

void LPQP::setEpsilonObjective(double eps_obj)
{
    _eps_obj = eps_obj;
}

void LPQP::setInitialLPActive(bool initial_lp_active)
{
    _initial_lp_active = initial_lp_active;
}

void LPQP::setInitialLPImprovmentRatio(double initial_lp_improvement_ratio)
{
    _initial_lp_improvement_ratio = initial_lp_improvement_ratio;
}

void LPQP::setInitialLPRhoStart(double initial_lp_rho_start)
{
    _initial_lp_rho_start = initial_lp_rho_start;
}

void LPQP::setInitialRhoSimilarValues(bool initial_rho_similar_values)
{
    _initial_rho_similar_values = initial_rho_similar_values;
}

void LPQP::setInitialRhoFactorKLSmaller(double initial_rho_factor_kl_smaller)
{
    _initial_rho_factor_kl_smaller = initial_rho_factor_kl_smaller;
}

void LPQP::setEpsilonMP(double eps)
{
    _eps_mp = eps;
}

void LPQP::setMaximumNumberOfIterationsMP(size_t num_max_iter)
{
    _num_max_iter_mp = num_max_iter;
}

void LPQP::setSkipIfIncrease(bool skip_if_increase)
{
    _skip_if_increase = skip_if_increase;
}

std::vector<double>& LPQP::getHistoryObjective()
{
    return _history_obj;
}

std::vector<double>& LPQP::getHistoryObjectiveQP()
{
    return _history_obj_qp;
}

std::vector<double>& LPQP::getHistoryObjectiveLP()
{
    return _history_obj_lp;
}

std::vector<double>& LPQP::getHistoryObjectiveDecoded()
{
    return _history_obj_decoded;
}

std::vector<size_t>& LPQP::getHistoryIteration()
{
    return _history_iter;
}

std::vector<double>& LPQP::getHistoryRho()
{
    return _history_rho;
}

double LPQP::log0(double x)
{
    assert(x >= 0);

    if (x == 0) {
        return 1.0;
    }
    else {
        return log(x);
    }
}
