#include <TreeInference.h>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

namespace {

class TreeInferenceTest : public ::testing::Test {
};

TEST_F(TreeInferenceTest, RandomSmall)
{
    size_t num_states = 2;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(5);
    for (size_t i=0; i<5; i++) {
        theta_unary[i].setRandom(num_states);
    }
    
    std::vector< Eigen::VectorXd > theta_pair;
    theta_pair.resize(4);
    for (size_t i=0; i<4; i++) {
        theta_pair[i].setRandom(num_states*num_states);
    }

    Eigen::MatrixXi edges;
    edges.setZero(2, 4);

    edges(0,0) = 0;
    edges(1,0) = 1;
    
    edges(0,1) = 1;
    edges(1,1) = 2;
    
    edges(0,2) = 1;
    edges(1,2) = 3;
    
    edges(0,3) = 4;
    edges(1,3) = 3;

    double beta = 1;
    TreeInference inf(theta_unary, theta_pair, edges);
    inf.run(beta);
}

TEST_F(TreeInferenceTest, BurglarAlarm)
{
    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(5);
    
    std::vector< Eigen::VectorXd > theta_pair;
    theta_pair.resize(4);

    Eigen::MatrixXi edges;
    edges.setZero(2, 4);

    // burglar
    theta_unary[0].setZero(2);
    theta_unary[0](0) = log(0.99);
    theta_unary[0](1) = log(0.01);
    
    // burglar, sprinkler
    edges(0,0) = 0;
    edges(1,0) = 1;
    theta_pair[0].setZero(4);
    theta_pair[0](0) = log(0.5);
    theta_pair[0](1) = log(0.1);
    theta_pair[0](2) = log(0.5);
    theta_pair[0](3) = log(0.9);
    
    // burglar, alarm 
    edges(0,1) = 0;
    edges(1,1) = 2;
    theta_pair[1].setZero(4);
    theta_pair[1](0) = log(0.99);
    theta_pair[1](1) = log(0.1);
    theta_pair[1](2) = log(0.01);
    theta_pair[1](3) = log(0.9);
    
    // sprinkler, grass
    edges(0,2) = 1;
    edges(1,2) = 3;
    theta_pair[2].setZero(4);
    theta_pair[2](0) = log(0.7);
    theta_pair[2](1) = log(0.4);
    theta_pair[2](2) = log(0.3);
    theta_pair[2](3) = log(0.6);
    
    // sprinkler, dog
    edges(0,3) = 1;
    edges(1,3) = 4;
    theta_pair[3].setZero(4);
    theta_pair[3](0) = log(0.8);
    theta_pair[3](1) = log(0.3);
    theta_pair[3](2) = log(0.2);
    theta_pair[3](3) = log(0.7);
    
    // sprinkler has no unary
    theta_unary[1].setZero(2);
    theta_unary[1](0) = 0;
    theta_unary[1](1) = 0;
    
    // observe evidence (alarm, grass, dog): clamp variables
    theta_unary[2].setZero(2);
    theta_unary[2](0) = -1e6;
    theta_unary[2](1) = 0;
    
    theta_unary[3].setZero(2);
    theta_unary[3](0) = -1e6;
    theta_unary[3](1) = 0;
    
    theta_unary[4].setZero(2);
    theta_unary[4](0) = 0;
    theta_unary[4](1) = -1e6;

    double beta = 1.0;
    TreeInference inf(theta_unary, theta_pair, edges);
    inf.run(beta);
    
    // check marginal for burglar not being present
    ASSERT_NEAR( inf.getUnaryMarginals()[0][0], 0.55, 0.01);

    // check logZ
    ASSERT_NEAR( exp(inf.getLogPartitionSum()), 0.003753, 0.00001);
}


} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
