#include <LPQPSDD.h>
#include <SmoothDualDecomposition.h>
#include <SmoothDualDecompositionFistaDescent.h>
#include <SmoothDualDecompositionLBFGS.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>

LPQPSDD::LPQPSDD(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges,
                            const std::vector< std::vector<size_t> >& decomposition,
                            Solver solver
                            ):
    LPQP(theta_unary, theta_pair, edges)
{
    // set the smooth dual decomposition solver
    if (solver == FISTADESCENT) {
        _sdd = new SmoothDualDecompositionFistaDescent(_theta_unary, _theta_pair, _edges, decomposition);
    }
    else if (solver == LBFGS) {
        _sdd = new SmoothDualDecompositionLBFGS(_theta_unary, _theta_pair, _edges, decomposition);
    }
    else {
        printf("Invalid solver!\n");
        assert(0);
    }
    _sdd->setShowProgress(false);
    _sdd->setMaximumNumberOfIterations(_num_max_iter_mp);
    _sdd->setEpsilonGradientNorm(_eps_mp);

    // determine the weight of a node in the concave part of the DC objective
    _node_weight_concave.setZero(_num_vertices);
    for (size_t d_idx=0; d_idx<decomposition.size(); d_idx++) {
        Eigen::VectorXd node_weight_cur;
        node_weight_cur.setZero(_num_vertices);
        for (size_t e_idx=0; e_idx<decomposition[d_idx].size(); e_idx++) {
            size_t e_idx_original = decomposition[d_idx][e_idx];
            size_t i = edges(0, e_idx_original);
            size_t j = edges(1, e_idx_original);
            if (node_weight_cur(i)<1) {
                node_weight_cur(i) = 1;
            }
            if (node_weight_cur(j)<1) {
                node_weight_cur(j) = 1;
            }
        }
        _node_weight_concave += node_weight_cur;
    }
    _node_weight_concave /= static_cast<double>(decomposition.size());

    // each edge has a different weight for the KL term
    _edge_weight_kl.setZero(_num_edges);
    for (size_t d_idx=0; d_idx<decomposition.size(); d_idx++) {
        Eigen::VectorXd edge_weight_cur;
        edge_weight_cur.setZero(_num_edges);
        for (size_t e_idx=0; e_idx<decomposition[d_idx].size(); e_idx++) {
            size_t e_idx_original = decomposition[d_idx][e_idx];
            if (edge_weight_cur(e_idx_original)<1) {
                edge_weight_cur(e_idx_original) = 1;
            }
        }
        _edge_weight_kl += edge_weight_cur;
    }
    _edge_weight_kl /= static_cast<double>(decomposition.size());
}

LPQPSDD::~LPQPSDD()
{
    delete _sdd;
}

void LPQPSDD::run()
{
    double rho = _rho_start;

    // solve smoothed linear program for smaller and smaller rho (i.e.,
    // problem we solve becomes more and more the LP). If no sufficient
    // decrease in *QP* objective anymore we stop and return the final rho
    if (_initial_lp_active) {
        rho = runInitialLP();
        rho = _rho_start;
    }

    // if solution with initial rho is already integer, there is no reason to
    // run the LPQP algorithm
    double entropy = computeUnaryEntropy();
    entropy /= static_cast<double>(_num_vertices);
    if (entropy < _eps_entropy) {
        printf("Unary entropy very small. Stopping.\n");
        return;
    }

    // ensure that rho*KL and LP term have roughly a similar value with the
    // initial solution, if not change rho such that this is the case
    if (_initial_rho_similar_values) {
        double rho_new = determineInitialRho();
        if (rho_new > rho) {
            rho = rho_new;
        }
    }
    printf("Starting with the following rho: %f.\n", rho);
    
    // LPQP algorithm
    printProgressHeader();
    size_t iter_total = 0;
    while (rho < _rho_end) {
        double obj_old = std::numeric_limits<double>::max();
        for (size_t iter_dc=0; iter_dc<_num_max_iter_dc; iter_dc++) {
            // modify the unary potential according to concave part in DC
            for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
                double c = _node_weight_concave(v_idx);
                //// add the true value
                //_theta_unary_curr[v_idx].array() = 
                //    _theta_unary[v_idx].array() - 
                //    rho*c*_mu_unary[v_idx].array().log();

                // alternative: only add the difference, with min element
                // not being updated at all
                Eigen::VectorXd temp = -rho*c*_mu_unary[v_idx].array().log();
                temp.array() -= temp.minCoeff();
                _theta_unary_curr[v_idx] = _theta_unary[v_idx] + temp;
            }

            // update unary potentials & re-run
            _sdd->setThetaUnary(_theta_unary_curr);
            size_t num_iter_sdd = _sdd->run(rho);

            // update marginals (TODO: speedup by not copying all the time!)
            _mu_unary = _sdd->getUnaryMarginals();
            _mu_pair = _sdd->getPairwiseMarginals();

            // show the progress
            iter_total++;
            if (iter_total % 20 == 0) {
                printf("\n");
                printProgressHeader();
            }
            double obj = computeObjective(rho);
            // objective should not increase, if it does, the algorithm proably either
            // converged or we are in a poor numerical range, so increase rho
            if (obj_old < obj) {
                printf("Objective increased! Old value: %f, new value: %f.\n", obj_old, obj);
                if (_skip_if_increase) {
                    break;
                }
            }
            // absolute convergence in objective?
            // TODO: investigate relative convergence instead!
            if (std::abs(obj-obj_old) < _eps_obj) {
                break;
            }
            obj_old = obj;
            double obj_qp = computeQPValue();
            double obj_lp = computeLPValue();
            double obj_decoded = computeObjectiveDecoded();
            printProgress(rho, iter_dc+1, num_iter_sdd, obj, obj_qp, obj_lp, obj_decoded);

            // history logging
            _history_obj.push_back(obj);
            _history_obj_qp.push_back(obj_qp);
            _history_obj_lp.push_back(obj_lp);
            _history_obj_decoded.push_back(obj_decoded);
            _history_iter.push_back(num_iter_sdd);
            _history_rho.push_back(rho);

            // TODO: check convergence of the marginals
 
            // convergence check: if average node entropy is below a threshold, we
            // assume we converged
            double entropy = computeUnaryEntropy();
            entropy /= static_cast<double>(_num_vertices);
            if (entropy < _eps_entropy) {
                printf("Unary entropy very small. Stopping.\n");
                return;
            }
        }

        //// TODO: investigate this convergence check
        //// KL-divergence convergence check
        //double dkl = computeKLDivergence();
        //dkl /= static_cast<double>(_num_edges);
        //if (dkl < _eps_dkl) {
        //    printf("KL divergence below threshold. Stopping.\n");
        //    return;
        //}

        // rho-scheduling
        // TODO: investigate different rho scheduling schemes! Similar to
        // deterministic annealing etc.
        rho *= _rho_schedule_constant;
    }
}

void LPQPSDD::setEpsilonMP(double eps)
{
    _eps_mp = eps;
    _sdd->setEpsilonGradientNorm(eps);
}

void LPQPSDD::setMaximumNumberOfIterationsMP(size_t num_max_iter)
{
    _num_max_iter_mp = num_max_iter;
    _sdd->setMaximumNumberOfIterations(num_max_iter);
}

double LPQPSDD::computeKLDivergence()
{
    double d = 0;

    // Kullback-Leibler divergence between pairwise and unary marginals
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        double dkl = 0;
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        for (size_t ki=0; ki<_num_states[i]; ki++) {
            assert(_mu_unary[i][ki]>=0);
            for (size_t kj=0; kj<_num_states[j]; kj++) {
                assert(_mu_unary[j][kj]>=0);
                assert(_mu_pair[e_idx](ki+_num_states[i]*kj)>=0);
                dkl += _mu_pair[e_idx](ki+_num_states[i]*kj)*log0(_mu_pair[e_idx](ki+_num_states[i]*kj));
                dkl -= _mu_pair[e_idx](ki+_num_states[i]*kj)*log0(_mu_unary[i](ki)*_mu_unary[j](kj));
            }
        }
        d += dkl*_edge_weight_kl(e_idx);
    }

    //if (d < 0) {
    //    printf("Warning! LPQP::computeKLDivergence, d is < 0, d == %f\n", d);
    //    d = 0;
    //}

    return d;
}

double LPQPSDD::runInitialLP()
{
    // TODO: consider increasing maximum number of iterations and epsilon for
    // decreasing rho! (i.e. make it faster for the less important iterations)
    
    // TODO: needs a rework

    printf("\nInitial LP solver for decreasing values of rho.\n");
    printf("%10s | %7s | %10s | %10s | %10s\n", "rho", "MP iter", "QP obj", "KL obj", "dec obj");

    double obj_qp_old = std::numeric_limits<double>::max();
    double rho = _initial_lp_rho_start;
    while (rho > _rho_start) {
        size_t num_iter_sdd = _sdd->run(rho);
        
        // TODO: don't copy!
        _mu_unary = _sdd->getUnaryMarginals();
        _mu_pair = _sdd->getPairwiseMarginals();

        // print progress
        double obj_qp = computeQPValue();
        double d_kl = computeKLDivergence();
        double obj_decoded = computeObjectiveDecoded();
        printf("%10.6g   %7d   %10.6g   %10.6g   %10.6g\n", rho, \
            static_cast<int>(num_iter_sdd), obj_qp, d_kl,
            obj_decoded);

        // ensure that there is a sufficient decrease in the QP objective!
        if ( ((obj_qp_old-obj_qp)/std::abs(obj_qp) < _initial_lp_improvement_ratio)  ){
            printf("Not a large enough decrease, switching to LPQP!\n");
            break;
        }
        obj_qp_old = obj_qp;
        
        rho /= 2;
    }

    printf("Initial LP solver finished.\n\n");

    return rho;
}
