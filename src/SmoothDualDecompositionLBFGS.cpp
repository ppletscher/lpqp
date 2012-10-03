#include <SmoothDualDecompositionLBFGS.h>
#include <TreeInference.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <limits>
#include <lbfgs.h>

SmoothDualDecompositionLBFGS::SmoothDualDecompositionLBFGS(
        std::vector< Eigen::VectorXd >& theta_unary,
        std::vector< Eigen::VectorXd >& theta_pair,
        const Eigen::MatrixXi& edges,
        const std::vector< std::vector<size_t> >& decomposition,
        bool do_pruning):
    SmoothDualDecomposition(theta_unary, theta_pair, edges, decomposition, do_pruning)
{
    initializeDualVariable(_lambda);
}

size_t SmoothDualDecompositionLBFGS::run(double rho)
{
    _rho = rho;

    // allocate memory for liblbfgs
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *_m_x = lbfgs_malloc(_num_elements_dual);
    if (_m_x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.");
        return 1;
    }

    for (size_t i=0; i<_num_elements_dual; i++) {
        _m_x[i] = 0;
    }

    // options for liblbfgs
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.m = 50;
    param.max_iterations = _max_num_iterations;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO; // another linesearch resulted in nans
    
    _num_evals = 0;

    int ret = lbfgs(_num_elements_dual, _m_x, &fx, _evaluate, _progress, this, &param);

    // report the result
    if (ret < 0) {
        printf("L-BFGS optimization terminated with status code = %d.\n", ret);
    }
    
    // copy results to the solution
    _lambda.setZero(_num_elements_dual);
    for (size_t i=0; i<_num_elements_dual; i++) {
        _lambda[i] = _m_x[i];
    }
    
    // get marginals from lambda
    computeObjective(rho, _lambda);
    setMarginalsUnary();
    setMarginalsPair();

    if (_m_x != NULL) {
        lbfgs_free(_m_x);
        _m_x = NULL;
    }

    // return the number of iterations
    return _num_evals;
}

lbfgsfloatval_t SmoothDualDecompositionLBFGS::evaluate(
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    // TODO: don't copy things around but rather use a map to the memory!

    // convert weight vector from liblbfgs format to Eigen
    _lambda.setZero(n);
    for (size_t i = 0; i < (size_t) n; i++) {
        assert(!std::isnan(x[i]));
        _lambda[i] = x[i];
    }
        
    // evaluation (change sign as we minimize)
    Eigen::VectorXd grad;
    lbfgsfloatval_t fx;
    fx = -computeObjectiveAndGradient(_rho, grad, reinterpret_cast<const Eigen::VectorXd&>(_lambda));

    // convert gradient vector from Eigen format to liblbfgs
    for (size_t i = 0; i < (size_t)n; i++) {
        g[i] = -grad[i]; // change sign as LBFGS minimizes!
    }

    _num_evals++;

    return fx;
}
