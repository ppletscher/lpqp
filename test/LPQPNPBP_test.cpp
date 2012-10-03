#include <LPQPNPBP.h>
#include <vector>
#include <Eigen/Core>

#include <gtest/gtest.h>

namespace {

class GridTest : public ::testing::Test {
};

TEST_F(GridTest, FirstAndOnly)
{
    size_t M = 10;
    size_t N = 10;
    size_t num_states = 2;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(M*N);
    for (size_t i=0; i<M*N; i++) {
        theta_unary[i].setRandom(num_states);
    }

    std::vector< Eigen::VectorXd > theta_pair;
    theta_pair.resize((M-1)*N+(N-1)*M);
    for (size_t e_idx=0; e_idx<(M-1)*N+(N-1)*M; e_idx++) {
        theta_pair[e_idx].setRandom(num_states*num_states);
        theta_pair[e_idx] = theta_pair[e_idx].array()*2;
    }
    Eigen::MatrixXi edges;
    edges.setZero(2, (M-1)*N+(N-1)*M);

    size_t idx=0;
    for (size_t j=0; j<N; j++) {
        for (size_t i=0; i<M; i++) {
            if (i < M-1) {
                edges(0,idx) = i+M*j;
                edges(1,idx) = (i+1)+M*j;
                idx++;
            }
            if (j < N-1) {
                edges(0,idx) = i+M*j;
                edges(1,idx) = i+M*(j+1);
                idx++;
            }
        }
    }

    LPQPNPBP lpqp(theta_unary, theta_pair, edges);

    lpqp.run();

    // TODO: do some actual unit testing!
}

TEST_F(GridTest, OnlyOneEdge)
{
    size_t M = 2;
    size_t N = 2;
    size_t num_states = 2;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(M*N);
    for (size_t i=0; i<M*N; i++) {
        theta_unary[i].setRandom(num_states);
    }
    
    std::vector< Eigen::VectorXd > theta_pair;
    theta_pair.resize(1);
    theta_pair[0].setRandom(num_states*num_states);
    theta_pair[0] = theta_pair[0].array()*2;
    Eigen::MatrixXi edges;
    edges.setZero(2, 1);
    edges(0,0) = 0;
    edges(1,0) = 1;

    LPQPNPBP lpqp(theta_unary, theta_pair, edges);

    lpqp.run();

    // TODO: do some actual unit testing!
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
