#ifndef __DEFINED_GRAPHDECOMPOSITION_H
#define __DEFINED_GRAPHDECOMPOSITION_H

#include <vector>
#include <Eigen/Core>

class GraphDecomposition {

private:
    // edge list, matrix of dimension 2xnum_edges
    const Eigen::MatrixXi _edges;

    size_t _num_edges;
    size_t _num_vertices;

    // graph is stored as a list of neighbors for each variable
    std::vector< std::vector< std::pair<size_t,size_t> > > _neighbors;

    // actual decomposition methods
    // TODO: should probably try to merge these methods, as the implementation
    // is almost the same, except for a few details
    std::vector< std::vector<size_t> > decompose_with_stack();
    std::vector< std::vector<size_t> > decompose_with_queue();

public:
    GraphDecomposition(Eigen::MatrixXi& edges);

    enum DecompositionType{DECOMPOSITION_STACK, DECOMPOSITION_QUEUE};

    std::vector< std::vector<size_t> > decompose(DecompositionType type = DECOMPOSITION_STACK);
    void pprint(std::vector< std::vector<size_t> > &decomposition);
};

#endif
