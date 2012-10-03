#ifndef __DEFINED_SMOOTHDUALDECOMPOSITIONGENERAL_H
#define __DEFINED_SMOOTHDUALDECOMPOSITIONGENERAL_H

#include <vector>
#include <Eigen/Core>
#include <TreeInference.h>

// in order to efficiently lookup in which decompositions a edge/vertex occurs
struct GraphElementMapping {
    size_t tree_idx;
    size_t element_idx; // in tree
    size_t dualvar_idx;
};


// Each decomposition contains a subset of the edges and vertices from the
// original graph. The resulting graph is assumed to contain no cycles,
// connectivity is not assumed however.

class SmoothDualDecomposition {

protected:
	
    // unary and pairwise potentials. Goal is to minimize!
    // NOTE: these have a different sign than the theta in TreeInference!!!
    std::vector< Eigen::VectorXd > _theta_unary; // dim: num_states*num_vertices
    std::vector< Eigen::VectorXd > _theta_pair; // dim: num_states^2*num_edges

    // edge list, matrix of dimension 2xnum_edges
    const Eigen::MatrixXi _edges;

    size_t _num_vertices;
    size_t _num_edges;
    std::vector< size_t > _num_states;

    // a decomposition is stored as a list of edges. The collection of all
    // decompositions is simply a vector of descompositions. 
    std::vector< std::vector<size_t> > _decomposition_node;
    std::vector< std::vector<size_t> > _decomposition_edge;
    std::vector< TreeInference* > _tree;

    std::vector< std::vector<GraphElementMapping> > _node_lookup;
    std::vector< std::vector<GraphElementMapping> > _edge_lookup;

    std::vector< Eigen::VectorXd > _mu_unary;
    std::vector< Eigen::VectorXd > _mu_pair;
    
    size_t _max_num_iterations;
    double _eps_gnorm;

    bool _show_progress;

    bool _do_pruning;
    std::vector< std::vector<GraphElementMapping> > _node_lookup_original;
    std::vector< std::vector<GraphElementMapping> > _edge_lookup_original;
    
    size_t _num_elements_dual;

    std::vector<size_t> _disconnected_nodes;
    
    void initializeNodeDecomposition();
    void initializeLookupTables();
    void initializeTrees();
    void initializeDualVariable(Eigen::VectorXd& lambda);

    int findNodeIDinTree(size_t original_node, size_t t_idx);

    void setMarginalsUnary();
    void setMarginalsPair();

    void setUnariesFromLambda(size_t decomposition_idx, const Eigen::VectorXd& lambda);
    void setPairwiseFromLambda(size_t decomposition_idx, const Eigen::VectorXd& lambda);

    double computeObjectiveAndGradient(double rho, Eigen::VectorXd&
                gradient, const Eigen::VectorXd& lambda);
    double computeObjective(double rho, const Eigen::VectorXd& lambda);
    
    void printProgressHeader();
    void printProgress(size_t k, double fx, double gnorm, double step);

    void addContributionToGradient(size_t idx_original,
                size_t decomposition_idx, const Eigen::VectorXd& nu,
                Eigen::VectorXd& gradient,
                const std::vector< std::vector<GraphElementMapping> >& lookup,
                const std::vector< Eigen::VectorXd >& theta);

    void pruneLookupTables();

    size_t initializeLookupTable(
        std::vector< std::vector<GraphElementMapping> >& lookup,
        std::vector< std::vector<size_t> >& decomposition,
        std::vector< Eigen::VectorXd >& theta,
        size_t dualvar_idx);
    
public:
    
    SmoothDualDecomposition(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            const Eigen::MatrixXi& edges,
                            const std::vector< std::vector<size_t> >& decomposition,
                            bool do_pruning=true);
    
    virtual ~SmoothDualDecomposition();

    virtual size_t run(double rho=1.0) = 0;

    std::vector< Eigen::VectorXd >& getUnaryMarginals();

    std::vector< Eigen::VectorXd >& getPairwiseMarginals();

    void setMaximumNumberOfIterations(size_t max_num_iter);
    void setEpsilonGradientNorm(double eps_gnorm);

    void setThetaUnary(std::vector< Eigen::VectorXd >& theta_unary);
    void setShowProgress(bool show_progress);

    //double adaptRho(const Eigen::VectorXd& lambda,
    //        double rho_start, double rho_end, double epsilon);
};

#endif
