#include <TRWS.h>
#include <LPQPSDD.h>
#include <GraphDecomposition.h>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

namespace {

class LPQPSDDDecompositionTest : public ::testing::Test {
	public:
		std::vector< std::vector<size_t> > get_decomposition(size_t type, size_t M, size_t N, Eigen::MatrixXi& edges);
};

// type 0 is the grid decomposition
std::vector< std::vector<size_t> > LPQPSDDDecompositionTest::get_decomposition(size_t type, size_t M, size_t N, Eigen::MatrixXi& edges)
{
	std::vector< std::vector<size_t> > decomposition;

	if(type == 0){
		decomposition.resize(2);

		size_t e_idx = 0;
		for (size_t j=0; j<N; j++) {
			for (size_t i=0; i<M; i++) {
				// vertical edge
				if (i<M-1) {
					decomposition[0].push_back(e_idx);
					e_idx++;
				}
				// horizontal edge
				if (j<N-1) {
					decomposition[1].push_back(e_idx);
					e_idx++;
				}
			}
		}
	}
	else if(type == 1){
	    GraphDecomposition gd(edges);
	    GraphDecomposition::DecompositionType dt = GraphDecomposition::DECOMPOSITION_STACK;
        return gd.decompose(dt);
	}

	else if(type == 2){
        GraphDecomposition gd(edges);
        GraphDecomposition::DecompositionType dt = GraphDecomposition::DECOMPOSITION_QUEUE;
        return gd.decompose(dt);
	}

	return decomposition;
}

TEST_F(LPQPSDDDecompositionTest, Grid)
{
    size_t num_states = 3;
    size_t M = 5;
    size_t N = 5;

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
	TRWS trws(theta_unary, theta_pair, edges);
	trws.run();

    //get the grid decomposition
    size_t type = 0;
    std::vector< std::vector<size_t> > grid_decomposition = get_decomposition(type, M, N, edges);

    // decomposition returned by the GraphDecomposition
    type = 1; //stack
    std::vector< std::vector<size_t> > stack_decomposition = get_decomposition(type, M, N, edges);
    type = 2; //queue
    std::vector< std::vector<size_t> > queue_decomposition = get_decomposition(type, M, N, edges);

    //print decompositions
    //GraphDecomposition gd(edges);
    //gd.pprint(grid_decomposition);
    //gd.pprint(stack_decomposition);
    //gd.pprint(queue_decomposition);
    //return;

    clock_t start, stop;

    double rho = 1e-2;
    double rho_end = 1;

    while (rho <= rho_end) {

    	std::cout << "rho is " << rho << std::endl;

		LPQPSDD lpqp_grid(theta_unary, theta_pair, edges, grid_decomposition);
		LPQPSDD lpqp_stack(theta_unary, theta_pair, edges, stack_decomposition);
		LPQPSDD lpqp_queue(theta_unary, theta_pair, edges, queue_decomposition);

		lpqp_grid.setRhoStart(rho);
		lpqp_stack.setRhoStart(rho);
		lpqp_queue.setRhoStart(rho);

		double time_grid, time_stack, time_queue;

		std::cout << "run grid " << rho << std::endl;
		start = clock();
		lpqp_grid.run();
		stop = clock();
		time_grid = (double) (stop-start)/CLOCKS_PER_SEC;

		std::cout << "run stack " << rho << std::endl;
		start = clock();
		lpqp_stack.run();
        stop = clock();
        time_stack = (double) (stop-start)/CLOCKS_PER_SEC;

		std::cout << "run queue " << rho << std::endl;
		start = clock();
		lpqp_queue.run();
        stop = clock();
        time_queue = (double) (stop-start)/CLOCKS_PER_SEC;

		std::vector< Eigen::VectorXd > mu_before_rounding = lpqp_grid.getUnaryMarginals();
		double qp_val_before_rounding = lpqp_grid.computeQPValue();
		double lp_val_before_rounding = lpqp_grid.computeLPValue();
		lpqp_grid.roundSolution();
		double qp_val_after_rounding = lpqp_grid.computeQPValue();
		double lp_val_after_rounding = lpqp_grid.computeLPValue();
		std::cout << "Grid: run time " << time_grid << std::endl;
		std::cout << "Grid: qp value before is " << qp_val_before_rounding << " after " << qp_val_after_rounding << std::endl;
		std::cout << "Grid: lp value before is " << lp_val_before_rounding << " after " << lp_val_after_rounding << std::endl;

		mu_before_rounding = lpqp_stack.getUnaryMarginals();
		qp_val_before_rounding = lpqp_stack.computeQPValue();
		lp_val_before_rounding = lpqp_stack.computeLPValue();
		lpqp_stack.roundSolution();
		qp_val_after_rounding = lpqp_stack.computeQPValue();
		lp_val_after_rounding = lpqp_stack.computeLPValue();
		std::cout << "stack: run time " << time_stack << std::endl;
		std::cout << "stack: qp value before is " << qp_val_before_rounding << " after " << qp_val_after_rounding << std::endl;
		std::cout << "stack: lp value before is " << lp_val_before_rounding << " after " << lp_val_after_rounding << std::endl;

		mu_before_rounding = lpqp_queue.getUnaryMarginals();
        qp_val_before_rounding = lpqp_queue.computeQPValue();
        lp_val_before_rounding = lpqp_queue.computeLPValue();
        lpqp_queue.roundSolution();
        qp_val_after_rounding = lpqp_queue.computeQPValue();
        lp_val_after_rounding = lpqp_queue.computeLPValue();
        std::cout << "Queue: run time " << time_queue << std::endl;
        std::cout << "queue: qp value before is " << qp_val_before_rounding << " after " << qp_val_after_rounding << std::endl;
        std::cout << "queue: lp value before is " << lp_val_before_rounding << " after " << lp_val_after_rounding << std::endl;

		// check that the marginals are similar
		for (size_t i=0; i<theta_unary.size(); i++) {
			std::cout << "unary: "<< i << std::endl;
			for (size_t j = 0; j < num_states; j ++){
				std::cout << lpqp_grid.getUnaryMarginals()[i][j] << ", " << lpqp_stack.getUnaryMarginals()[i][j] << ", " << lpqp_queue.getUnaryMarginals()[i][j] << std::endl;
			}
		}

		rho *= 10;
    }
}

} // namespace



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
