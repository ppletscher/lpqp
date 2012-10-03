#include <LPQPNPBP.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>

// TODO: optionally log the marginals in history after each iteration, not only the rounded solution
// TODO: store the best state *ever* found! (and return this)
// TODO: speedup: only run through graph once to compute the different objectives
// TODO: speedup: do not update messages for nodes that already converged!

LPQPNPBP::LPQPNPBP(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges):
    LPQP(theta_unary, theta_pair, edges),
    _msg_to_first(),
    _msg_to_second()
{
    // initializes messages passed for each edge
    _msg_to_first.resize(_num_edges);
    _msg_to_second.resize(_num_edges);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _msg_to_first[e_idx].setZero(_num_states[i]);
        _msg_to_second[e_idx].setZero(_num_states[j]);
    }
}

double LPQPNPBP::logsumexp(const Eigen::VectorXd& x)
{
    double m = x.maxCoeff();

    Eigen::VectorXd a;
    a = (x.array()-m).array().exp();
    double s = a.sum();
    return log(s)+m;
}

void LPQPNPBP::normalizeVector(Eigen::VectorXd& v)
{
    double logZ = logsumexp(v);
    v = v.array() - logZ;
}

Eigen::VectorXd LPQPNPBP::collectIncomingAndUnary(size_t i, double beta)
{
    size_t num_neighbors = _neighbors[i].size();

    Eigen::VectorXd m;
    m.setZero(_num_states[i]);

    // unary potential
    m = -_theta_unary_curr[i];

    // incoming messages
    for (size_t n_idx=0; n_idx<num_neighbors; n_idx++) {
        size_t e = _neighbors[i][n_idx].second;

        if (i == static_cast<size_t>(_edges(0,e))) {
            m += _msg_to_first[e];
        }
        else {
            m += _msg_to_second[e];
        }
    }

    // for nodes that are disconnected from the rest of the graph
    // num_neighbors would be zero, so make sure we do not divide by zero!
    if (num_neighbors > 0) {
        m *= 1.0/(beta*num_neighbors);
    }
    
    return m;
}

double LPQPNPBP::sendMessages(size_t i, double beta)
{
    size_t num_neighbors = _neighbors[i].size();
    double change = 0.0;

    Eigen::VectorXd m = collectIncomingAndUnary(i, beta);

    // send messages out
    for (size_t n_idx=0; n_idx<num_neighbors; n_idx++) {
        // send a message from node i to node j
        size_t j = _neighbors[i][n_idx].first;
        size_t e = _neighbors[i][n_idx].second;

        Eigen::VectorXd m_cur = m;
        // sum-product
        if (i == static_cast<size_t>(_edges(0,e))) {
            // i is the first index, so store message in _msg_to_second[e]
            Eigen::VectorXd m_old = _msg_to_second[e];
            m_cur -= _msg_to_first[e]*1.0/beta;

            Eigen::Map<Eigen::MatrixXd> theta_edge(_theta_pair[e].data(),_num_states[i],_num_states[j]);

            for (size_t kj=0; kj<_num_states[j]; kj++) {
                Eigen::VectorXd m_temp;
                m_temp = (-theta_edge.col(kj).array())*(1.0/beta);
                m_temp += m_cur;
                _msg_to_second[e](kj) = logsumexp(m_temp);
            }

            // exponentiate and normalize
            _msg_to_second[e] *= beta;
            normalizeVector(_msg_to_second[e]);
            change += (m_old-_msg_to_second[e]).norm();
        }
        else {
            // j is the first index, so store message in _msg_to_first[e]
            Eigen::VectorXd m_old = _msg_to_first[e];
            m_cur -= _msg_to_second[e]*1.0/beta;
            
            Eigen::Map<Eigen::MatrixXd> theta_edge(_theta_pair[e].data(),_num_states[j],_num_states[i]);
            
            for (size_t kj=0; kj<_num_states[j]; kj++) {
                Eigen::VectorXd m_temp;
                m_temp = (-theta_edge.row(kj).transpose().array())*(1.0/beta);
                m_temp += m_cur;                
                _msg_to_first[e](kj) = logsumexp(m_temp);
            }

            // exponentiate and normalize
            _msg_to_first[e] *= beta;
            normalizeVector(_msg_to_first[e]);
            change += (m_old-_msg_to_first[e]).norm();
        }
    }

    return change;
}

void LPQPNPBP::checkVectorNotNan(Eigen::VectorXd& m, size_t idx)
{
    for (size_t k=0; k<static_cast<size_t>(m.size()); k++) {
        if (std::isnan(m[k])) {
            printf("vector of with idx %d contains a nan!\n", static_cast<int>(idx));
            return;
        }
    }
}

size_t LPQPNPBP::passMessages(double beta)
{
    double change;
    size_t iter_mp;

    for (iter_mp=0; iter_mp<_num_max_iter_mp; iter_mp++) {
        // TODO: option to permute the indices to send the messages in a random order

        change = 0;
        for (size_t i=0; i<_num_vertices; i++) {
            change += sendMessages(i, beta);
        }
        change /= static_cast<double>(_num_vertices);

        // TODO: additional options for tracking progress.
        // change currently tracks the average change of the messages,
        // one should investigate whether max change or some other statistics
        // leads to better results. Additionally, one could instead track the
        // actual change of the marginals (instead of messages), the change of
        // marginals can be tracked as follows:
        // change = updateMarginalsUnary(beta);
        // in this case we however probably also need to change the constants
        // for convergence in the main loop.

        if (change < _eps_mp) {
            break;
        }
    }

    return iter_mp+1;
}

double LPQPNPBP::updateMarginalsUnary(double beta)
{
    double change = 0;

    for (size_t i=0; i<_num_vertices; i++) {
        Eigen::VectorXd m_old = _mu_unary[i];

        _mu_unary[i] = collectIncomingAndUnary(i, beta);

        // normalize
        normalizeVector(_mu_unary[i]);
        _mu_unary[i] = _mu_unary[i].array().exp();

        change += (_mu_unary[i]-m_old).norm();
    }
    change /= static_cast<double>(_num_vertices);

    return change;
}

void LPQPNPBP::updateMarginalsPair(double beta)
{
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = _edges(0,e_idx);
        size_t j = _edges(1,e_idx);

        // product of incoming messages at i
        Eigen::VectorXd mi = collectIncomingAndUnary(i, beta);

        // product of incoming messages at j
        Eigen::VectorXd mj = collectIncomingAndUnary(j, beta);
        
        // divide by message in the opposite direction
        if (i == static_cast<size_t>(_edges(0,e_idx))) {
            mi -= _msg_to_first[e_idx]*1.0/(beta);
            mj -= _msg_to_second[e_idx]*1.0/(beta);
        }
        else {
            mi -= _msg_to_second[e_idx]*1.0/(beta);
            mj -= _msg_to_first[e_idx]*1.0/(beta);
        }

        // incorporate pairwise potential
        for (size_t kj=0; kj<_num_states[j]; kj++) {
            for (size_t ki=0; ki<_num_states[i]; ki++) {
                double a = (-_theta_pair[e_idx](ki+kj*_num_states[i]))*(1.0/beta)+mi(ki)+mj(kj);
                _mu_pair[e_idx](ki+kj*_num_states[i]) = a;
            }
        }

        normalizeVector(_mu_pair[e_idx]);
        _mu_pair[e_idx] = _mu_pair[e_idx].array().exp();
    }
}

void LPQPNPBP::run()
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
                //// add the true value
                //_theta_unary_curr[v_idx].array() = 
                //    _theta_unary[v_idx].array() - 
                //    rho*_neighbors[v_idx].size()*_mu_unary[v_idx].array().log();

                // alternative: only add the difference, with min element
                // not being updated at all
                Eigen::VectorXd temp = -rho*_neighbors[v_idx].size()*_mu_unary[v_idx].array().log();
                temp.array() -= temp.minCoeff();
                _theta_unary_curr[v_idx] = _theta_unary[v_idx] + temp;
            }

            // message passing
            size_t num_iter_mp = passMessages(rho);

            // compute the new marginals
            double change_marginal = 0;
            change_marginal += updateMarginalsUnary(rho);
            updateMarginalsPair(rho);

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
            printProgress(rho, iter_dc+1, num_iter_mp, obj, obj_qp, obj_lp, obj_decoded);
            
            // history logging
            _history_obj.push_back(obj);
            _history_obj_qp.push_back(obj_qp);
            _history_obj_lp.push_back(obj_lp);
            _history_obj_decoded.push_back(obj_decoded);
            _history_iter.push_back(num_iter_mp);
            _history_rho.push_back(rho);

            // TODO: investigate convergence checks for the marginals
            //// convergence check (only consider the unary marginals)
            //if (change_marginal < _eps_dc) {
            //    break;
            //}
            
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

double LPQPNPBP::computeKLDivergence()
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
        d += dkl;
    }

    if (d < 0) {
        printf("Warning! LPQP::computeKLDivergence, d is < 0, d == %f\n", d);
        d = 0;
    }

    return d;
}

double LPQPNPBP::runInitialLP()
{
    // TODO: consider increasing maximum number of iterations and epsilon for
    // decreasing rho! (i.e. make it faster for the less important iterations)
    
    // TODO: needs a rework

    printf("\nInitial LP solver for decreasing values of rho.\n");
    printf("%10s | %7s | %10s | %10s | %10s\n", "rho", "MP iter", "QP obj", "KL obj", "dec obj");

    double obj_qp_old = std::numeric_limits<double>::max();
    double rho = _initial_lp_rho_start;
    while (rho > _rho_start) {
        size_t num_iter_mp = passMessages(rho);
        
        updateMarginalsUnary(rho);
        updateMarginalsPair(rho);

        // print progress
        double obj_qp = computeQPValue();
        double d_kl = computeKLDivergence();
        double obj_decoded = computeObjectiveDecoded();
        printf("%10.6g   %7d   %10.6g   %10.6g   %10.6g\n", rho, \
            static_cast<int>(num_iter_mp), obj_qp, d_kl,
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
