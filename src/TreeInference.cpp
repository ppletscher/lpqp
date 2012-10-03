#include <TreeInference.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <stack>
#include <cmath>
#include <Eigen/Core>

TreeInference::TreeInference(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges):
    _theta_unary(theta_unary),
    _theta_pair(theta_pair),
    _edges(edges),
    _beta(1.0),
    _num_vertices(theta_unary.size()),
    _num_edges(theta_pair.size()),
    _num_states(),
    _neighbors(),
    _mu_unary(),
    _mu_pair(),
    _msg_to_first(),
    _msg_to_second(),
    _order()
{
    // input checks
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
    _msg_to_first.resize(_num_edges);
    _msg_to_second.resize(_num_edges);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _mu_pair[e_idx].setConstant(_num_states[i]*_num_states[j], 1.0/static_cast<double>(_num_states[i]*_num_states[j]));

        _msg_to_first[e_idx].setZero(_num_states[i]);
        _msg_to_second[e_idx].setZero(_num_states[j]);
    }
    
    // initialize the graph data structure from the edges
    _neighbors.resize(_num_vertices);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _neighbors[i].push_back(std::make_pair(j, e_idx));
        _neighbors[j].push_back(std::make_pair(i, e_idx));
    }

    // topological sort for determining order in which messages should be sent
    _order.setZero(2, _num_edges);
    determineOrder();
}


void TreeInference::determineOrder()
{
    size_t num_visited = 0;

    // container for all the unvisited variables in the factor graph
    std::set<size_t> not_visited;
    for (size_t i=0; i<_num_vertices; i++) {
        not_visited.insert(i);
    }
    
    // flag for each variable whether it has been visited already
    std::vector<bool> has_been_visited;
    has_been_visited.resize(_num_vertices, false);

    // for each disconnected component perform a DFS
    // stores variable plus the factor from which the DFS is coming from
    size_t order_idx = 0;
    std::stack< std::pair<size_t,size_t> > variables_to_visit;
    while (num_visited < _num_vertices) {
        assert(variables_to_visit.empty());
        size_t var;
        int var_coming_from;

        // pick an unvisited variable
        var = *(not_visited.begin());
        var_coming_from = -1;
        variables_to_visit.push( std::make_pair(var, var_coming_from) );

        // depth-first-search
        while(!variables_to_visit.empty()) {
            // get a new variable to visit
            std::pair<size_t, size_t > v;
            v = variables_to_visit.top();
            variables_to_visit.pop();
            var = v.first;
            var_coming_from = v.second;

            // cycle detection
            assert(!has_been_visited[var]);

            // mark variable as visited
            has_been_visited[var] = true;
            not_visited.erase(not_visited.find(var));
            num_visited++;

            // visit all the neighboring variables
            for (size_t n_idx=0; n_idx<_neighbors[var].size(); n_idx++) {
                size_t var_neighbor = _neighbors[var][n_idx].first;
                size_t e_idx = _neighbors[var][n_idx].second;
                
                // don't go "up"
                if (static_cast<int>(var_neighbor) == var_coming_from) {
                    continue;
                }
                    
                // add neighboring variables to the stack
                variables_to_visit.push( std::make_pair(var_neighbor, var) );
                
                // store in the schedule (second entry in order refers to
                // whether i to j or j to i, 0 means according to _edges convention)
                _order(0,order_idx) = e_idx;
                _order(1,order_idx) = !(_edges(0,e_idx)==static_cast<int>(var));
                order_idx++;
            }
        }
    }
}

void TreeInference::setMarginalsUnary()
{
    for (size_t i=0; i<_num_vertices; i++) {
        _mu_unary[i] = collectIncomingAndUnary(i);
        _mu_unary[i] *= _beta;
        normalizeVector(_mu_unary[i]);
        _mu_unary[i] = _mu_unary[i].array().exp();
    }
}

void TreeInference::setMarginalsPair()
{
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0,e_idx);
        size_t j = _edges(1,e_idx);

        // product of incoming messages at i
        Eigen::VectorXd mi = collectIncomingAndUnary(i);

        // product of incoming messages at j
        Eigen::VectorXd mj = collectIncomingAndUnary(j);
        
        // divide by message in the opposite direction
        if (i == static_cast<size_t>(_edges(0,e_idx))) {
            mi -= _msg_to_first[e_idx];
            mj -= _msg_to_second[e_idx];
        }
        else {
            mi -= _msg_to_second[e_idx];
            mj -= _msg_to_first[e_idx];
        }

        // incorporate pairwise potential
        for (size_t kj=0; kj<_num_states[j]; kj++) {
            for (size_t ki=0; ki<_num_states[i]; ki++) {
                double a = _theta_pair[e_idx](ki+kj*_num_states[i])+mi(ki)+mj(kj);
                a *= _beta;
                _mu_pair[e_idx](ki+kj*_num_states[i]) = a;
            }
        }

        normalizeVector(_mu_pair[e_idx]);
        _mu_pair[e_idx] = _mu_pair[e_idx].array().exp();
    }
}

void TreeInference::run(double beta)
{
    _beta = beta;

    // pass messages up
    for (int idx=_order.cols()-1; idx>=0; idx--) {
        size_t e_idx = _order(0,idx);
        bool direction = !_order(1,idx);
        passMessage(e_idx, direction);
    }
    
    // pass messages up
    for (int idx=0; idx<_order.cols(); idx++) {
        size_t e_idx = _order(0,idx);
        bool direction = _order(1,idx);
        passMessage(e_idx, direction);
    }

    // compute marginals
    setMarginalsUnary();
    setMarginalsPair();

    // entropy, average energy and log partition sum
    computeEntropy();
    computeAverageEnergy();

    _free_energy = _average_energy - (1.0/_beta)*_entropy;
    _logZ = -_beta*(_free_energy);
}
    
void TreeInference::passMessage(size_t e_idx, bool direction)
{
    size_t i, j;
    if (!direction) {
        i = _edges(0,e_idx);
        j = _edges(1,e_idx);
    }
    else {
        i = _edges(1,e_idx);
        j = _edges(0,e_idx);
    }

    Eigen::VectorXd m = collectIncomingAndUnary(i, static_cast<int>(j));

    // sum-product (divide by incoming in same direction)
    if (!direction) {
        Eigen::Map<Eigen::MatrixXd> theta_edge(_theta_pair[e_idx].data(),_num_states[i],_num_states[j]);

        for (size_t kj=0; kj<_num_states[j]; kj++) {
            Eigen::VectorXd m_temp;
            m_temp = theta_edge.col(kj).array()*_beta;
            m_temp += m*_beta;
            _msg_to_second[e_idx](kj) = logsumexp(m_temp);
        }

        // exponentiate and normalize
        _msg_to_second[e_idx] /= _beta;
        normalizeVector(_msg_to_second[e_idx]);
    }
    else {
        Eigen::Map<Eigen::MatrixXd> theta_edge(_theta_pair[e_idx].data(),_num_states[j],_num_states[i]);
        
        for (size_t kj=0; kj<_num_states[j]; kj++) {
            Eigen::VectorXd m_temp;
            m_temp = theta_edge.row(kj).transpose().array()*_beta;
            m_temp += m*_beta;                
            _msg_to_first[e_idx](kj) = logsumexp(m_temp);
        }

        // exponentiate and normalize
        _msg_to_first[e_idx] /= _beta;
        normalizeVector(_msg_to_first[e_idx]);
    }
}

Eigen::VectorXd TreeInference::collectIncomingAndUnary(size_t i, int exclude)
{
    size_t num_neighbors = _neighbors[i].size();

    Eigen::VectorXd m;

    // unary potential
    m = _theta_unary[i];

    // incoming messages
    for (size_t n_idx=0; n_idx<num_neighbors; n_idx++) {
        if (static_cast<int>(_neighbors[i][n_idx].first) == exclude) {
            continue;
        }
        size_t e = _neighbors[i][n_idx].second;

        if (i == static_cast<size_t>(_edges(0,e))) {
            m += _msg_to_first[e];
        }
        else {
            m += _msg_to_second[e];
        }
    }
    
    return m;
}

double TreeInference::logsumexp(const Eigen::VectorXd& x)
{
    double m = x.maxCoeff();

    Eigen::VectorXd a;
    a = (x.array()-m).array().exp();
    double s = a.sum();
    return log(s)+m;
}

void TreeInference::normalizeVector(Eigen::VectorXd& v)
{
    double logZ = logsumexp(v);
    v = v.array() - logZ;
}

std::vector< Eigen::VectorXd >& TreeInference::getUnaryMarginals()
{
    return _mu_unary;
}

std::vector< Eigen::VectorXd >& TreeInference::getPairwiseMarginals()
{
    return _mu_pair;
}

void TreeInference::computeEntropy()
{
    // unary
    double entropy_unary = 0;
    for (size_t i=0; i<_num_vertices; i++) {
        double d = _neighbors[i].size();
        double H = 0;
        for (size_t k=0; k<_num_states[i]; k++) {
            H -= _mu_unary[i](k)*log0(_mu_unary[i](k));
        }

        entropy_unary += (1-d)*H;
    }

    // pairwise
    double entropy_pair = 0;
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        double H = 0;
        for (size_t ki=0; ki<_num_states[i]; ki++) {
            for (size_t kj=0; kj<_num_states[j]; kj++) {
                H -= _mu_pair[e_idx](ki+_num_states[i]*kj)*log0(_mu_pair[e_idx](ki+_num_states[i]*kj));
            }
        }

        entropy_pair += H;
    }
    
    _entropy = entropy_unary + entropy_pair;
}

void TreeInference::computeAverageEnergy()
{
    // NOTE: energy has opposite sign than what we have in the thetas,
    // therefore we need to negate the value here!

    _average_energy = 0;

    // unary
    for (size_t i=0; i<_num_vertices; i++) {
        for (size_t k=0; k<_num_states[i]; k++) {
            assert(_mu_unary[i](k) >= 0);
            _average_energy -= _mu_unary[i](k)*_theta_unary[i](k);
        }
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        for (size_t ki=0; ki<_num_states[i]; ki++) {
            for (size_t kj=0; kj<_num_states[j]; kj++) {
                assert(_mu_pair[e_idx](ki+_num_states[i]*kj) >=0);
                _average_energy -= _mu_pair[e_idx](ki+_num_states[i]*kj)*_theta_pair[e_idx](ki+_num_states[i]*kj);
            }
        }
    }
}

double TreeInference::getLogPartitionSum()
{
    return _logZ;
}

void TreeInference::setThetaUnary(size_t i, Eigen::VectorXd& theta)
{
    for (int k=0; k<theta.size(); k++) {
        assert(!std::isnan(theta(k)));
    }

    _theta_unary[i] = theta;
}

void TreeInference::setThetaPair(size_t e, Eigen::VectorXd& theta)
{
    for (int k=0; k<theta.size(); k++) {
        assert(!std::isnan(theta(k)));
    }

    _theta_pair[e] = theta;
}

double TreeInference::getFreeEnergy()
{
    return _free_energy;
}

void TreeInference::runExhaustive(double beta)
{
    _beta = beta;
    
    size_t num_configs = 1;
    for (size_t idx=0; idx<_num_vertices; idx++) {
        num_configs *= _num_states[idx];
    }
    
    // initialize unary marginals
    _mu_unary.resize(_num_vertices);
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        _mu_unary[v_idx].setConstant(_num_states[v_idx], -1e10);
    }

    // initialize pairwise marginals and messages (duals of marginals)
    _mu_pair.resize(_num_edges);
    _msg_to_first.resize(_num_edges);
    _msg_to_second.resize(_num_edges);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        _mu_pair[e_idx].setConstant(_num_states[i]*_num_states[j], -1e10);
    }

    Eigen::VectorXi label;
    label.setZero(_num_vertices);

    Eigen::VectorXd logz;
    logz.setZero(2);
    logz[0] = -1e10;
    
    for (size_t idx=0; idx<num_configs; idx++) {
        incrementLabel(label);
        double s = computeScore(label);
        logz[1] = s;
        logz[0] = logsumexp(logz);
        addScoreToMarginals(s, label);
    }
    _logZ = logz[0];
    _free_energy = -(1.0/beta)*_logZ;

    // unary
    for (size_t i=0; i<_num_vertices; i++) {
        _mu_unary[i] = _mu_unary[i].array() - logsumexp(_mu_unary[i]);
        _mu_unary[i] = _mu_unary[i].array().exp();
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        _mu_pair[e_idx] = _mu_pair[e_idx].array() - logsumexp(logz);
        _mu_pair[e_idx] = _mu_pair[e_idx].array().exp();
    }
}

double TreeInference::computeScore(const Eigen::VectorXi& label)
{
    double score = 0;

    // unaries
    for (size_t i=0; i<_num_vertices; i++) {
        score += _theta_unary[i][label[i]]*_beta;
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        score += _theta_pair[e_idx][label[i]+_num_states[i]*label[j]]*_beta;
    }

    return score;
}

void TreeInference::incrementLabel(Eigen::VectorXi& label)
{
    for (size_t i=0; i<_num_vertices; i++) {
        size_t r = label[i]+1;
        label[i] = r % _num_states[i];
        if (r<_num_states[i]){
            break;
        }
    }
}

void TreeInference::addScoreToMarginals(double s, const Eigen::VectorXi& label)
{
    Eigen::VectorXd logz;
    logz.setZero(2);
    logz[1] = s;

    // unary
    for (size_t i=0; i<_num_vertices; i++) {
        logz[0] = _mu_unary[i][label[i]];
        _mu_unary[i][label[i]] = logsumexp(logz);
    }

    // pairwise
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0, e_idx);
        size_t j = _edges(1, e_idx);
        logz[0] = _mu_pair[e_idx][label[i]+label[j]*_num_states[i]];
        _mu_pair[e_idx][label[i]+label[j]*_num_states[i]] = logsumexp(logz);
    }
}

double TreeInference::log0(double x)
{
    assert(x >= 0);

    if (x == 0) {
        return 1.0;
    }
    else {
        return log(x);
    }
}


void TreeInference::checkPotentials()
{
    // vertices
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        for (int k=0; k<_theta_unary[v_idx].size(); k++) {
            assert(!std::isnan(_theta_unary[v_idx](k)));
        }
    }

    // edges
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        for (int k=0; k<_theta_pair[e_idx].size(); k++) {
            assert(!std::isnan(_theta_pair[e_idx](k)));
        }
    }
}
