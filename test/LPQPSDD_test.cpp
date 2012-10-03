#include <LPQP.h>
#include <TRWS.h>
#include <LPQPSDD.h>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

namespace {

class LPQPSDDTest : public ::testing::Test {
};

TEST_F(LPQPSDDTest, Grid)
{
    size_t num_states = 4;
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
    
    // run TRWS for the same problem
    TRWS trws(theta_unary, theta_pair, edges);
    trws.run();

    LPQP* lpqp1;
    lpqp1 = new LPQPSDD(theta_unary, theta_pair, edges, decomposition);
    delete lpqp1;

    LPQPSDD lpqp_fista(theta_unary, theta_pair, edges, decomposition, LPQPSDD::FISTADESCENT);
    lpqp_fista.setRhoStart(1e-1);
    lpqp_fista.run();
    std::vector< Eigen::VectorXd > mu_before_rounding = lpqp_fista.getUnaryMarginals();
    lpqp_fista.roundSolution();
    
    LPQPSDD lpqp_lbfgs(theta_unary, theta_pair, edges, decomposition, LPQPSDD::LBFGS);
    lpqp_lbfgs.setRhoStart(1e-1);
    lpqp_lbfgs.run();
    std::vector< Eigen::VectorXd > mu_before_rounding_lbfgs = lpqp_lbfgs.getUnaryMarginals();
    lpqp_lbfgs.roundSolution();

    // check that the marginals are similar
    for (size_t i=0; i<theta_unary.size(); i++) {
        std::cout << "unary: "<< i << std::endl;
        for (size_t j = 0; j < num_states; j ++){
            std::cout << mu_before_rounding[i][j] << ", " << lpqp_fista.getUnaryMarginals()[i][j] << ", " << mu_before_rounding_lbfgs[i][j] << ", " << lpqp_lbfgs.getUnaryMarginals()[i][j] << ", " << trws.getUnaryMarginals()[i][j] << std::endl;
        }
    }
    
    // TODO: do some actual unit testing!
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
