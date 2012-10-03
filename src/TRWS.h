#ifndef __DEFINED_TRWS_H
#define __DEFINED_TRWS_H

#include <vector>
#include <Eigen/Core>

class TRWS {

private:
    // unary and pairwise potentials. NOTE: this is for *minimization*.
    std::vector< Eigen::VectorXd > _theta_unary; // dim: num_states*num_vertices
    std::vector< Eigen::VectorXd > _theta_pair; // dim: num_states^2*num_edges

    // edge list, matrix of dimension 2xnum_edges
    const Eigen::MatrixXi _edges;

    size_t _num_max_iter;

    size_t _num_vertices;
    size_t _num_edges;
    std::vector< size_t > _num_states;

    std::vector< Eigen::VectorXd > _mu_unary;
    std::vector< Eigen::VectorXd > _mu_pair;

public:
    TRWS(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges);

    void run();

    std::vector< Eigen::VectorXd >& getUnaryMarginals();

    std::vector< Eigen::VectorXd >& getPairwiseMarginals();

    void setMaximumNumberOfIterations(size_t num_max_iter);
};

#endif
