#include <SmoothDualDecompositionFistaDescent.h>
#include <TreeInference.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <limits>

SmoothDualDecompositionFistaDescent::SmoothDualDecompositionFistaDescent(
        std::vector< Eigen::VectorXd >& theta_unary,
        std::vector< Eigen::VectorXd >& theta_pair,
        const Eigen::MatrixXi& edges,
        const std::vector< std::vector<size_t> >& decomposition,
        bool do_pruning):
    SmoothDualDecomposition(theta_unary, theta_pair, edges, decomposition, do_pruning),
    _lipschitz_constant_optimistic(2),
    _lipschitz_inc_d(2),
    _lipschitz_inc_u(2)
{
    initializeDualVariable(_lambda);
    initializeDualVariable(_y_old);
    initializeDualVariable(_v);
    initializeDualVariable(_u);
    initializeDualVariable(_y);
}

size_t SmoothDualDecompositionFistaDescent::run(double rho)
{
    //double rho_end = rho;
    //rho = 10;

    // step length
    double omega;
    omega = _lipschitz_constant_optimistic;
    
    // objectives
    double obj_lambda, obj_y_old;
    obj_y_old = -std::numeric_limits<double>::max();

    // gradient
    Eigen::VectorXd gradient;
    double gnorm = std::numeric_limits<double>::max();
    
    //// TODO: should possibly consider pruning all the labels that have an
    //// inf in theta, but this would get quite messy! Could use the property
    //// if a var is once inf it stays inf in the whole LPQP framework.
    //// TODO: also investigate how much slower the computations get because
    //// of the infs!
    //size_t num_inf = 0;
    //for (size_t v_idx=0; v_idx<_num_vertices; v_idx++) {
    //    for (size_t k=0; k<_num_states[v_idx]; k++) {
    //        if (std::isinf(_theta_unary[v_idx][k])) {
    //            num_inf++;
    //        }
    //    }
    //}
    //std::cout << "num_inf: " << num_inf << std::endl;

    size_t iter = 0;	
    while(!stoppingCriteriaMet(iter, gnorm)){

        //// TODO: play around with this! Seems to work, but we need to improve this!
        //if ((iter % 10) == 0) {
        //    printf("rho before: %f.\n", rho);
        //    rho = adaptRho(_lambda, rho, rho_end, 1e-1);
        //    printf("changed rho: %f.\n", rho);
        //    obj_y_old = computeObjective(rho, _y);
        //}

        _y_old = _y;

        double obj_new;
        obj_new = gradientStep(rho, _lambda, obj_lambda, gradient, gnorm, _u, omega);

        printProgress(iter, obj_new, gnorm, 1.0/omega);
        
        if (obj_new > obj_y_old) {
            _y = _u;
        }
        else {
            _y = _y_old;
        }
        double theta = 2/(static_cast<double>(iter)+2.0);
        _v =  _y_old+1.0/theta*(_u-_y_old);
        theta = 2/(static_cast<double>(iter)+3.0);
        _lambda = (1-theta)*_y+theta*_v;
        obj_y_old = obj_new;
        
        iter++;
	}

    // _y is the final solution 
    _lambda = _y;

    // due to the line search we have to call inference
    // again before returning to actually get the latest marginals of the
    // final value of lambda!
    obj_lambda = computeObjective(rho, _lambda);
    setMarginalsUnary();
    setMarginalsPair();

    return iter;
}

double SmoothDualDecompositionFistaDescent::gradientStep(double rho,
            const Eigen::VectorXd& lambda,
            double& obj_lambda,
            Eigen::VectorXd& gradient,
            double& gradient_norm_squared,
            Eigen::VectorXd& y, double& omega)
{
    double obj_new;
    double obj_approx;
    
    // evaluate objective, gradient and gradient norm
    obj_lambda = computeObjectiveAndGradient(rho, gradient, lambda);
    gradient_norm_squared = gradient.squaredNorm();

    // backtracking
    do {
        y = lambda + gradient*(1.0/omega);
        obj_new = computeObjective(rho, y);
        obj_approx = obj_lambda+1/(2.0*omega)*gradient_norm_squared;
        if (obj_new < obj_approx) {
            omega *= _lipschitz_inc_u;
        }
    } while(obj_new < obj_approx);
    omega = std::max(_lipschitz_constant_optimistic, omega/_lipschitz_inc_d);

    return obj_new;
}

bool SmoothDualDecompositionFistaDescent::stoppingCriteriaMet(size_t iter, double gnorm)
{
	if ((gnorm < _eps_gnorm)){
        if (_show_progress) {
            printf("Norm of the gradient too small, stopping.\n");
        }
        return true;
    }
	if (iter >= _max_num_iterations){
        if (_show_progress) {
            printf("Maximum number of iterations reached.\n");
        }
		return true;
	}

    // TODO: investigate other checks, norm of solution differences, step length 
	
	return false;
}
