#include <SmoothDualDecomposition.h>
#include <TreeInference.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <set>
#include <Eigen/Core>

// TODO: this currently assumes 1/|T| as weight for each decomposition, but
// maybe we should allow for a vector with the weight of each decomposition
// (do not necessarily need to sum up to one!)

SmoothDualDecomposition::SmoothDualDecomposition(
        std::vector< Eigen::VectorXd >& theta_unary,
        std::vector< Eigen::VectorXd >& theta_pair,
        const Eigen::MatrixXi& edges,
        const std::vector< std::vector<size_t> >& decomposition_edge,
        bool do_pruning):
    _theta_unary(theta_unary),
    _theta_pair(theta_pair),
    _edges(edges),
    _num_vertices(theta_unary.size()),
    _num_edges(theta_pair.size()),
    _num_states(),
    _decomposition_edge(decomposition_edge),
    _tree(),
    _mu_unary(),
    _mu_pair(),
    _max_num_iterations(1000),
    _eps_gnorm(1e-4),
    _show_progress(true),
    _do_pruning(do_pruning),
    _num_elements_dual()
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

    // initialize pairwise marginals 
    _mu_pair.resize(_num_edges);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);

        _mu_pair[e_idx].setConstant(_num_states[i]*_num_states[j], 1.0/static_cast<double>(_num_states[i]*_num_states[j]));
    }

    // initialize data structures for dual variables & tree inference
    initializeNodeDecomposition();
    initializeLookupTables();
    initializeTrees();

    // assertion that all the edges covered at least once! (this also ensures
    // that all the nodes are covered once)
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        assert(_edge_lookup[e_idx].size() > 0);
    }
    
    // prune the dual variables that are in only one decomposition! We keep
    // the original data structure as this makes the computation of the
    // marginals much easier. This is however only called in the very end.
    if (_do_pruning) {
        _node_lookup_original = _node_lookup;
        _edge_lookup_original = _edge_lookup;
        pruneLookupTables();
    }

    // disconnected nodes are not part of any decomposition, carry out
    // computations for theses nodes separately
    std::vector<bool> node_is_disconnected;
    node_is_disconnected.resize(_num_vertices, true);
    for (size_t e_idx=0; e_idx<_num_edges; e_idx++) {
        size_t i = edges(0, e_idx);
        size_t j = edges(1, e_idx);
        node_is_disconnected[i] = false;
        node_is_disconnected[j] = false;
    }
    for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
        if (node_is_disconnected[v_idx]) {
            printf("Node %d is disconnected!\n", static_cast<int>(v_idx));
            _disconnected_nodes.push_back(v_idx);
        }
    }
}

SmoothDualDecomposition::~SmoothDualDecomposition()
{
    // the TreeInference objects are allocated dynamically, so make sure we
    // delete them again!
    for (size_t d_idx=0; d_idx<_tree.size(); d_idx++) {
        delete _tree[d_idx];
    }
}

void SmoothDualDecomposition::initializeNodeDecomposition()
{
    // initialize node decompositions from the edge decompositions
    // (essentially just check which nodes occurr in the tree)
    _decomposition_node.resize(_decomposition_edge.size());
    for (size_t d_idx=0; d_idx<_decomposition_edge.size(); d_idx++) {
        std::set<size_t> vars_in_d;
        for (size_t e_idx_in_d=0; e_idx_in_d<_decomposition_edge[d_idx].size(); e_idx_in_d++) {
            size_t e_idx_original = _decomposition_edge[d_idx][e_idx_in_d];
            vars_in_d.insert(_edges(0,e_idx_original));
            vars_in_d.insert(_edges(1,e_idx_original));
        }
        std::set<size_t>::iterator it;
        for (it=vars_in_d.begin(); it!=vars_in_d.end(); it++) {
            _decomposition_node[d_idx].push_back(*it);
        }
    }
}

size_t SmoothDualDecomposition::initializeLookupTable(
        std::vector< std::vector<GraphElementMapping> >& lookup,
        std::vector< std::vector<size_t> >& decomposition,
        std::vector< Eigen::VectorXd >& theta,
        size_t dualvar_idx)
{
    lookup.resize(theta.size());
    for (size_t d_idx=0; d_idx<decomposition.size(); d_idx++) {
        for (size_t element_idx_in_d=0; element_idx_in_d<decomposition[d_idx].size(); element_idx_in_d++) {
            size_t element_idx_original = decomposition[d_idx][element_idx_in_d];

            GraphElementMapping l;
            l.tree_idx = d_idx;
            l.element_idx = element_idx_in_d;
            l.dualvar_idx = dualvar_idx;
            lookup[element_idx_original].push_back(l);
            
            //std::cout << "d_idx: "<< d_idx
            //          << " v_idx_in_d: " << v_idx_in_d
            //          << " v_idx_original: " << v_idx_original
            //          << " dualvar: "  << dualvar_idx << std::endl;

            dualvar_idx += theta[element_idx_original].size();
        }
    }
    
    return dualvar_idx;
}

void SmoothDualDecomposition::initializeLookupTables()
{
    size_t dualvar_idx = 0;
    dualvar_idx = initializeLookupTable(_node_lookup, _decomposition_node, _theta_unary, dualvar_idx);
    dualvar_idx = initializeLookupTable(_edge_lookup, _decomposition_edge, _theta_pair, dualvar_idx);
    
    // number of elements in the dual vector
    _num_elements_dual = dualvar_idx;
}

void SmoothDualDecomposition::initializeTrees()
{
    _tree.clear();

    for (size_t d_idx=0; d_idx<_decomposition_edge.size(); d_idx++) {
        // nodes: simply initialize theta_unary to the right size
        size_t num_vertices = _decomposition_node[d_idx].size(); 
        std::vector< Eigen::VectorXd > theta_unary;
        theta_unary.resize(num_vertices);
        for (size_t i=0; i<num_vertices; i++) {
            theta_unary[i].setZero(_num_states[_decomposition_node[d_idx][i]]);
        }

        // edges: initalize theta_pair to the right size but also remap the
        // node indices in the edges (as some nodes might no longer occurr in
        // one of the trees)
        size_t num_edges = _decomposition_edge[d_idx].size();
        std::vector< Eigen::VectorXd > theta_pair;
        Eigen::MatrixXi edges;
        theta_pair.resize(num_edges);
        edges.setZero(2, num_edges);
        for (size_t e_idx=0; e_idx<num_edges; e_idx++) {
            size_t num_states = _theta_pair[_decomposition_edge[d_idx][e_idx]].size();
            theta_pair[e_idx].setZero(num_states);
            
            size_t i_original = _edges(0,_decomposition_edge[d_idx][e_idx]);
            size_t j_original = _edges(1,_decomposition_edge[d_idx][e_idx]);
            
            int i_new = findNodeIDinTree(i_original, d_idx);
            int j_new = findNodeIDinTree(j_original, d_idx);
            assert(i_new >= 0);
            assert(j_new >= 0);
            
            edges(0,e_idx) = static_cast<size_t>(i_new);
            edges(1,e_idx) = static_cast<size_t>(j_new);
            //std::cout << "i_original: " << i_original << std::endl;
            //std::cout << "j_original: " << j_original << std::endl;
            //std::cout << "i_new: " << i_new << std::endl;
            //std::cout << "j_new: " << j_new << std::endl;
        }

        TreeInference* inf;
        inf = new TreeInference(theta_unary, theta_pair, edges);
        _tree.push_back(inf);
    }
}

int SmoothDualDecomposition::findNodeIDinTree(size_t original_node, size_t t_idx)
{
    int node_idx_in_tree = -1;

    for (size_t idx=0; idx<_node_lookup[original_node].size(); idx++) {
        if (_node_lookup[original_node][idx].tree_idx == t_idx) {
            node_idx_in_tree = _node_lookup[original_node][idx].element_idx;
            break;
        }
    }

    return node_idx_in_tree;
}

void SmoothDualDecomposition::initializeDualVariable(Eigen::VectorXd& lambda)
{
    lambda.setZero(_num_elements_dual);
}

// TODO: is there a way to combine the two functions below (setUnariesFromLambda and setPairwiseFromLambda)? They essentially do the same!

void SmoothDualDecomposition::setUnariesFromLambda(size_t decomposition_idx,
        const Eigen::VectorXd& lambda)
{
    // i loops over all the nodes in the decomposition decomposition_idx
    for (size_t i=0; i<_decomposition_node[decomposition_idx].size(); i++) {
        size_t i_original = _decomposition_node[decomposition_idx][i];

        // collect all the contributions from lambda
        Eigen::VectorXd l;
        l.setZero(_num_states[i_original]);
        for (size_t d_idx=0; d_idx<_node_lookup[i_original].size(); d_idx++) {
            GraphElementMapping m = _node_lookup[i_original][d_idx];
            size_t l_idx = m.dualvar_idx;
            if (decomposition_idx == m.tree_idx) {
                l += lambda.segment(l_idx, _num_states[i_original])*(_node_lookup[i_original].size()-1);
            }
            else {
                l -= lambda.segment(l_idx, _num_states[i_original]);
            }
        }
        if (_node_lookup[i_original].size() > 0) {
            l /= _node_lookup[i_original].size();
        }
    
        Eigen::VectorXd v = _theta_unary[i_original];
        if (_node_lookup[i_original].size() > 0) {
            v /= _node_lookup[i_original].size();
        }
        v += l;
        v = -v; // NOTE: negation as tree is maximizing!
        _tree[decomposition_idx]->setThetaUnary(i, v);
    }
}

void SmoothDualDecomposition::setPairwiseFromLambda(size_t decomposition_idx,
        const Eigen::VectorXd& lambda)
{
    // e_idx loops over all the edges in the decomposition decomposition_idx
    for (size_t e_idx=0; e_idx<_decomposition_edge[decomposition_idx].size(); e_idx++) {
        size_t e_original = _decomposition_edge[decomposition_idx][e_idx];

        // collect all the contributions from lambda
        Eigen::VectorXd l;
        size_t num_states = _theta_pair[e_original].size();
        l.setZero(num_states);
        for (size_t d_idx=0; d_idx<_edge_lookup[e_original].size(); d_idx++) {
            GraphElementMapping m = _edge_lookup[e_original][d_idx];
            size_t l_idx = m.dualvar_idx;
            if (decomposition_idx == m.tree_idx) {
                l += lambda.segment(l_idx, num_states)*(_edge_lookup[e_original].size()-1);
            }
            else {
                l -= lambda.segment(l_idx, num_states);
            }
        }
        if (_edge_lookup[e_original].size() > 0) {
            l /= _edge_lookup[e_original].size();
        }
    
        Eigen::VectorXd v = _theta_pair[e_original];
        if (_edge_lookup[e_original].size() > 0) {
            v /= _edge_lookup[e_original].size();
        }
        v += l;
        v = -v; // NOTE: negation as tree is maximizing!
        _tree[decomposition_idx]->setThetaPair(e_idx, v);
    }
}

std::vector< Eigen::VectorXd >& SmoothDualDecomposition::getUnaryMarginals()
{
    return _mu_unary;
}

std::vector< Eigen::VectorXd >& SmoothDualDecomposition::getPairwiseMarginals()
{
    return _mu_pair;
}

// TODO: is there a way to combine the two functions below (setMarginalsUnary and setMarginalsPair)? They essentially do the same!

void SmoothDualDecomposition::setMarginalsUnary()
{
    std::vector< std::vector<GraphElementMapping> >* node_lookup = &(_node_lookup);
    if (_do_pruning) {
        node_lookup = &_node_lookup_original;
    }

    // average of all the tree marginals sharing this node
    for (size_t v_idx=0; v_idx<(*node_lookup).size(); v_idx++) {
        _mu_unary[v_idx].setZero(_num_states[v_idx]);
        double c = 1.0/(static_cast<double>((*node_lookup)[v_idx].size()));
        for (size_t d_idx=0; d_idx<(*node_lookup)[v_idx].size(); d_idx++) {
            const GraphElementMapping& m = (*node_lookup)[v_idx][d_idx];
            _mu_unary[v_idx] += c*_tree[m.tree_idx]->getUnaryMarginals()[m.element_idx];
        }
    }

    // for disconnected nodes just choose the minimum
    for (size_t idx=0; idx<_disconnected_nodes.size(); idx++) {
        size_t v_idx = _disconnected_nodes[idx];
        size_t x;
        _theta_unary[v_idx].minCoeff(&x);
        _mu_unary[v_idx].setZero(_num_states[v_idx]);
        _mu_unary[v_idx][x] = 1;
    }
}

void SmoothDualDecomposition::setMarginalsPair()
{
    std::vector< std::vector<GraphElementMapping> >* edge_lookup = &(_edge_lookup);
    if (_do_pruning) {
        edge_lookup = &(_edge_lookup_original);
    }

    // average of all the tree marginals sharing this edge
    for (size_t e_idx=0; e_idx<(*edge_lookup).size(); e_idx++) {
        _mu_pair[e_idx].setZero(_theta_pair[e_idx].size());
        double c = 1.0/(static_cast<double>((*edge_lookup)[e_idx].size()));
        for (size_t d_idx=0; d_idx<(*edge_lookup)[e_idx].size(); d_idx++) {
            const GraphElementMapping& m = (*edge_lookup)[e_idx][d_idx];
            _mu_pair[e_idx] += c*_tree[m.tree_idx]->getPairwiseMarginals()[m.element_idx];
        }
    }
}

void SmoothDualDecomposition::addContributionToGradient(size_t idx_original,
                size_t decomposition_idx, const Eigen::VectorXd& nu,
                Eigen::VectorXd& gradient,
                const std::vector< std::vector<GraphElementMapping> >& lookup,
                const std::vector< Eigen::VectorXd >& theta)
{
    size_t num_states = theta[idx_original].size();

    for (size_t d_idx=0; d_idx<lookup[idx_original].size(); d_idx++) {
        const GraphElementMapping& m = lookup[idx_original][d_idx];
        if (m.tree_idx == decomposition_idx) {
            gradient.segment(m.dualvar_idx, num_states) += nu;
        }
        gradient.segment(m.dualvar_idx, num_states) -= nu*1/(lookup[idx_original].size());
    }
}

double SmoothDualDecomposition::computeObjectiveAndGradient(double rho,
            Eigen::VectorXd& gradient,
            const Eigen::VectorXd& lambda)
{
    //// is lambda valid?
    //for (int k=0; k<lambda.size(); k++) {
    //    assert(!isnan(lambda(k)));
    //}

    // initialize objective and gradient
    double objective = 0;
    gradient.setZero(_num_elements_dual);

    // compute objective and gradient
    for (size_t d_idx=0; d_idx<_tree.size(); d_idx++) {
        // update potentials of the different trees based on current lambda
        setUnariesFromLambda(d_idx, lambda);
        setPairwiseFromLambda(d_idx, lambda);

        // tree inference to get function value & marginals
        double c = 1.0/(static_cast<double>(_decomposition_node.size()));
        _tree[d_idx]->run(1.0/(rho*c));
        objective += _tree[d_idx]->getFreeEnergy();

        // gradient contributions of the marginals
        for (size_t node_idx=0; node_idx<_decomposition_node[d_idx].size(); node_idx++) {
            Eigen::VectorXd& v = _tree[d_idx]->getUnaryMarginals()[node_idx];
            size_t node_original = _decomposition_node[d_idx][node_idx];
            addContributionToGradient(node_original, d_idx, v, gradient,
                                        _node_lookup, _theta_unary);
        }
        for (size_t e_idx=0; e_idx<_decomposition_edge[d_idx].size(); e_idx++) {
            Eigen::VectorXd& v = _tree[d_idx]->getPairwiseMarginals()[e_idx];
            size_t edge_original = _decomposition_edge[d_idx][e_idx];
            addContributionToGradient(edge_original, d_idx, v, gradient,
                                        _edge_lookup, _theta_pair);
        }
    }

    // for disconnected nodes just choose the minimum
    for (size_t idx=0; idx<_disconnected_nodes.size(); idx++) {
        size_t v_idx = _disconnected_nodes[idx];
        objective += _theta_unary[v_idx].minCoeff();
    }
    
    //// valid gradient?
    //for (int k=0; k<gradient.size(); k++) {
    //    assert(!isnan(gradient(k)));
    //}

    return objective;
}

double SmoothDualDecomposition::computeObjective(double rho,
            const Eigen::VectorXd& lambda)
{
    //// valid lambda?
    //for (int k=0; k<lambda.size(); k++) {
    //    assert(!isnan(lambda(k)));
    //}

    // initialize objective
    double objective = 0;

    // compute objective
    for (size_t d_idx=0; d_idx<_tree.size(); d_idx++) {
        setUnariesFromLambda(d_idx, lambda);
        setPairwiseFromLambda(d_idx, lambda);

        // tree inference to get function value & marginals
        double c = 1.0/(static_cast<double>(_decomposition_node.size()));
        _tree[d_idx]->run(1.0/(c*rho));
        objective += _tree[d_idx]->getFreeEnergy();
    }
    
    // for disconnected nodes just choose the minimum
    for (size_t idx=0; idx<_disconnected_nodes.size(); idx++) {
        size_t v_idx = _disconnected_nodes[idx];
        objective += _theta_unary[v_idx].minCoeff();
    }

    return objective;
}

void SmoothDualDecomposition::setMaximumNumberOfIterations(size_t max_num_iter)
{
    _max_num_iterations = max_num_iter;
}

void SmoothDualDecomposition::setEpsilonGradientNorm(double eps_gnorm)
{
    _eps_gnorm = eps_gnorm;
}

void SmoothDualDecomposition::printProgressHeader()
{
    if (!_show_progress) {
        return;
    }
    printf("%5s | %20s | %10s | %10s\n", "iter", "objective", "gnorm", "step");
}

void SmoothDualDecomposition::printProgress(size_t k, double fx, double gnorm, double step)
{
    if (!_show_progress) {
        return;
    }
    if ((k % 20 == 0)) {
        printf("\n");
        printProgressHeader();
    }

    printf("%5d   %20.6g   %10.3e   %10.3e\n", static_cast<int>(k), fx, gnorm, step);
}

void SmoothDualDecomposition::setThetaUnary(std::vector< Eigen::VectorXd >& theta_unary)
{
    //// is theta_unary valid?
    //for (size_t i=0; i<theta_unary.size(); i++) {
    //    for (int k=0; k<theta_unary[i].size(); k++) {
    //        assert(!isnan(theta_unary[i](k)));
    //    }
    //}
    _theta_unary = theta_unary;
}

void SmoothDualDecomposition::setShowProgress(bool show_progress)
{
    _show_progress = show_progress;
}

void SmoothDualDecomposition::pruneLookupTables()
{
    printf("Pruning the dual variables.\n");
    printf("Originally we have %d dual variables.\n", static_cast<int>(_num_elements_dual));
    size_t dualvar_idx = 0;
    for (size_t v_idx=0; v_idx<_node_lookup.size(); v_idx++) {
        if (_node_lookup[v_idx].size() == 1) {
            _node_lookup[v_idx].clear();
            continue;
        }

        for (size_t d_idx=0; d_idx<_node_lookup[v_idx].size(); d_idx++) {
            _node_lookup[v_idx][d_idx].dualvar_idx = dualvar_idx;
            dualvar_idx += _num_states[v_idx];
        }
    }
    
    for (size_t e_idx=0; e_idx<_edge_lookup.size(); e_idx++) {
        if (_edge_lookup[e_idx].size() == 1) {
            _edge_lookup[e_idx].clear();
            continue;
        }

        for (size_t d_idx=0; d_idx<_edge_lookup[e_idx].size(); d_idx++) {
            _edge_lookup[e_idx][d_idx].dualvar_idx = dualvar_idx;
            dualvar_idx += _theta_pair[e_idx].size();
        }
    }
    
    // number of elements in the dual vector
    _num_elements_dual = dualvar_idx;
    
    printf("After pruning we have %d dual variables.\n", static_cast<int>(_num_elements_dual));
}

//// TODO: this needs to be improved, doesn't work yet!
//double SmoothDualDecomposition::adaptRho(const Eigen::VectorXd& lambda,
//            double rho_start, double rho_end, double epsilon)
//{
//    double rho = rho_start;
//
//    // ensure that the difference between the true objective and the
//    // smoothed objective for the current lambda is around epsilon
//    double obj_true = computeObjective(rho_end, lambda);
//    //std::cout << "obj_true: " << obj_true << std::endl;
//    double obj_smoothed = computeObjective(rho, lambda);
//    //std::cout << "obj_smoothed: " << obj_smoothed << std::endl;
//    while ((obj_true - obj_smoothed) > epsilon) {
//        rho /= 2.0;
//        obj_smoothed = computeObjective(rho, lambda);
//    }
//    rho = std::max(rho_end, rho);
//
//    return rho;
//}
