#include <GraphDecomposition.h>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

namespace {

class GraphDecompositionTest : public ::testing::Test {
};

TEST_F(GraphDecompositionTest, Grid)
{
    size_t M = 20;
    size_t N = 20;

    size_t num_edges = M*(N-1)+(M-1)*N;
    Eigen::MatrixXi edges;
    edges.setZero(2, num_edges);

    size_t e_idx = 0;
    for (size_t j=0; j<N; j++) {
        for (size_t i=0; i<M; i++) {
            // vertical edge
            if (i<M-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+1+j*M;
                e_idx++;
            }
            // horizontal edge
            if (j<N-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+(j+1)*M;
                e_idx++;
            }
        }
    }

    GraphDecomposition d = GraphDecomposition(edges);
    std::vector< std::vector<size_t> > decomposition = d.decompose();

    // check whether all the edges are covered
    std::vector<bool> covered_edge;
    covered_edge.resize(num_edges, false);
    for (size_t idx_d=0; idx_d<decomposition.size(); idx_d++) {
        for (size_t idx=0; idx<decomposition[idx_d].size(); idx++) {
            covered_edge[decomposition[idx_d][idx]] = true;
        }
    }
    for (size_t idx_e=0; idx_e<num_edges; idx_e++) {
        ASSERT_TRUE(covered_edge[idx_e]);
    }
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
