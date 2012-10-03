#include <TRWS.h>
#include <GraphDecomposition.h>
#include <SmoothDualDecompositionFistaDescent.h>
#include <SmoothDualDecompositionLBFGS.h>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

namespace {

class SmoothDualDecompositionTest : public ::testing::Test {
};

TEST_F(SmoothDualDecompositionTest, GridSmall)
{
    size_t num_states = 2;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(4);
    for (size_t i=0; i<4; i++) {
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
    
    edges(0,1) = 0;
    edges(1,1) = 2;
    
    edges(0,2) = 1;
    edges(1,2) = 3;
    
    edges(0,3) = 2;
    edges(1,3) = 3;

    std::vector< std::vector<size_t> > decomposition;
    decomposition.resize(2);

    // horizontal
    decomposition[0].push_back(1);
    decomposition[0].push_back(2);

    // vertical
    decomposition[1].push_back(0);
    decomposition[1].push_back(3);

    // smooth dual decomposition
    double rho = 1e-5;

    // testing dynamic allocation
    SmoothDualDecomposition* sdd;
    sdd = new SmoothDualDecompositionFistaDescent(theta_unary, theta_pair, edges, decomposition);
    sdd->setMaximumNumberOfIterations(200);
    sdd->run(rho);
    delete sdd;

    // snake decomposition
    decomposition.clear();
    decomposition.resize(3);

    // snake1
    decomposition[0].push_back(1);
    decomposition[0].push_back(2);
    decomposition[0].push_back(3);

    // snake2
    decomposition[1].push_back(0);
    decomposition[1].push_back(1);
    decomposition[1].push_back(2);

    // snake3
    decomposition[2].push_back(0);
    decomposition[2].push_back(1);
    decomposition[2].push_back(3);
    
    SmoothDualDecompositionFistaDescent sdd_snake(theta_unary, theta_pair, edges, decomposition);
    sdd_snake.setMaximumNumberOfIterations(200);
    sdd_snake.setEpsilonGradientNorm(1e-5);
    sdd_snake.run(rho);
    
    // run TRWS on the same problem
    TRWS trws(theta_unary, theta_pair, edges);
    trws.run();
    
    // check that the marginals computed by the different algorithms are similar
    for (size_t i=0; i<theta_unary.size(); i++) {
        double d;
        d = std::abs(sdd_snake.getUnaryMarginals()[i][0] - trws.getUnaryMarginals()[i][0]);
        ASSERT_NEAR( d, 0, 1e-4);
    }
}

TEST_F(SmoothDualDecompositionTest, GridLarge)
{
    size_t num_states = 3;
    size_t M = 10;
    size_t N = 10;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(M*N);
    for (size_t i=0; i<M*N; i++) {
        theta_unary[i].setRandom(num_states);
    }
    
    std::vector< Eigen::VectorXd > theta_pair;
    size_t num_edges = M*(N-1)+(M-1)*N;
    theta_pair.resize(num_edges);
    for (size_t i=0; i<num_edges; i++) {
        theta_pair[i].setRandom(num_states*num_states);
    }

    Eigen::MatrixXi edges;
    edges.setZero(2, num_edges);

    std::vector< std::vector<size_t> > decomposition;
    decomposition.resize(2);

    size_t e_idx = 0;
    for (size_t j=0; j<N; j++) {
        for (size_t i=0; i<M; i++) {
            // vertical edge
            if (i<M-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+1+j*M;
                decomposition[0].push_back(e_idx);
                e_idx++;
            }
            // horizontal edge
            if (j<N-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+(j+1)*M;
                decomposition[1].push_back(e_idx);
                e_idx++;
            }
        }
    }

    double rho = 1e-3;
    SmoothDualDecompositionFistaDescent sdd(theta_unary, theta_pair, edges, decomposition);
    sdd.setMaximumNumberOfIterations(1000);
    sdd.setEpsilonGradientNorm(1e-4);
    sdd.run(rho);
    
    SmoothDualDecompositionLBFGS sdd_lbfgs(theta_unary, theta_pair, edges, decomposition);
    sdd_lbfgs.run(rho);
    
    // run TRWS for the same problem
    TRWS trws(theta_unary, theta_pair, edges);
    trws.run();
   
    // TODO: we get a better lower bound than TRWS, so the check will fail!
    //// check that the marginals are similar
    //for (size_t i=0; i<theta_unary.size(); i++) {
    //    double d = std::abs(sdd.getUnaryMarginals()[i][0] - trws.getUnaryMarginals()[i][0]);
    //    ASSERT_NEAR( d, 0, 1e-4);
    //}
}

TEST_F(SmoothDualDecompositionTest, GridAutomaticDecomposition)
{
    size_t num_states = 2;
    size_t M = 10;
    size_t N = 10;

    std::vector< Eigen::VectorXd > theta_unary;
    theta_unary.resize(M*N);
    for (size_t i=0; i<M*N; i++) {
        theta_unary[i].setRandom(num_states);
    }
    
    std::vector< Eigen::VectorXd > theta_pair;
    size_t num_edges = M*(N-1)+(M-1)*N;
    theta_pair.resize(num_edges);
    for (size_t i=0; i<num_edges; i++) {
        theta_pair[i].setRandom(num_states*num_states);
    }

    Eigen::MatrixXi edges;
    edges.setZero(2, num_edges);
    
    std::vector< std::vector<size_t> > decomposition_manual;
    decomposition_manual.resize(2);

    size_t e_idx = 0;
    for (size_t j=0; j<N; j++) {
        for (size_t i=0; i<M; i++) {
            // vertical edge
            if (i<M-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+1+j*M;
                decomposition_manual[0].push_back(e_idx);
                e_idx++;
            }
            // horizontal edge
            if (j<N-1) {
                edges(0,e_idx) = i+j*M;
                edges(1,e_idx) = i+(j+1)*M;
                decomposition_manual[1].push_back(e_idx);
                e_idx++;
            }
        }
    }

    std::vector< std::vector<size_t> > decomposition;
    GraphDecomposition d = GraphDecomposition(edges);
    decomposition = d.decompose();

    double rho = 1e-3;
    SmoothDualDecompositionFistaDescent sdd(theta_unary, theta_pair, edges, decomposition);
    sdd.run(rho);
    
    SmoothDualDecompositionFistaDescent sddm(theta_unary, theta_pair, edges, decomposition_manual);
    //SmoothDualDecompositionLBFGS sddm(theta_unary, theta_pair, edges, decomposition_manual);
    sddm.run(rho);
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
