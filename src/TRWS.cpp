#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <MRFEnergy.h>
#include <minimize.h>
#include <ordering.h>
#include <treeProbabilities.h>
#include <TRWS.h>

// TODO: store the best state *ever* found! (and return this)

TRWS::TRWS(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges):
    _theta_unary(theta_unary),
    _theta_pair(theta_pair),
    _edges(edges),
    _num_max_iter(1000),
    _num_vertices(theta_unary.size()),
    _num_edges(theta_pair.size())
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
}

void TRWS::run()
{
    MRFEnergy<TypeGeneral>* energy;
    MRFEnergy<TypeGeneral>::NodeId* nodes;
	energy = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
    nodes = new MRFEnergy<TypeGeneral>::NodeId[_num_vertices];
    std::vector<TypeGeneral::REAL*> energy_terms;

    // unary costs
	for (size_t var = 0; var < _num_vertices; var++) {
        TypeGeneral::REAL* D = new TypeGeneral::REAL[_num_states[var]];
        
        for (size_t k = 0; k < _num_states[var]; k++) {
            D[k] = _theta_unary[var][k];
        }
    	
        nodes[var] = energy->AddNode(TypeGeneral::LocalSize(_num_states[var]), TypeGeneral::NodeData(D));
        energy_terms.push_back(D);
	}

    // pairwise costs
	for (size_t e = 0; e < _num_edges; e++) {
        size_t i = _edges(0,e);
        size_t j = _edges(1,e);

        TypeGeneral::REAL* D = new TypeGeneral::REAL[_num_states[i]*_num_states[j]];
        energy_terms.push_back(D);
        size_t K = _num_states[i]*_num_states[j];
        for (size_t idx=0; idx<K; idx++) {
                D[idx] = _theta_pair[e][idx];
        }                
        energy->AddEdge(nodes[(int)(i)], nodes[(int)(j)], TypeGeneral::EdgeData(TypeGeneral::GENERAL, D));
	}
    
    MRFEnergy<TypeGeneral>::Options options;
    options.m_iterMax = _num_max_iter;
    TypeGeneral::REAL energy_val, lowerBound;
    energy->Minimize_TRW_S(options, lowerBound, energy_val);

    // write results back to marginals array, which is assumed by our LPQP algorithms
	for (size_t var = 0; var < _num_vertices; var++) {
        _mu_unary[var].setZero(_num_states[var]);
        _mu_unary[var][energy->GetSolution(nodes[var])] = 1;
    }
    
    // free memory
    delete[] nodes;
	delete energy;
    for (size_t i=0; i<energy_terms.size(); i++) {
        delete[] energy_terms[i];
    }
}

std::vector< Eigen::VectorXd >& TRWS::getUnaryMarginals()
{
    return _mu_unary;
}

std::vector< Eigen::VectorXd >& TRWS::getPairwiseMarginals()
{
    return _mu_pair;
}

void TRWS::setMaximumNumberOfIterations(size_t num_max_iter)
{
    _num_max_iter = num_max_iter;
}
