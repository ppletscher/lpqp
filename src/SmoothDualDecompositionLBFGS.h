#ifndef __DEFINED_SMOOTHDUALDECOMPOSITIONLBFGS_H
#define __DEFINED_SMOOTHDUALDECOMPOSITIONLBFGS_H

#include <vector>
#include <Eigen/Core>
#include <TreeInference.h>
#include <SmoothDualDecomposition.h>
#include <lbfgs.h>

class SmoothDualDecompositionLBFGS : public SmoothDualDecomposition {
private:
    // dual variables
    Eigen::VectorXd _lambda;
    
    lbfgsfloatval_t *_m_x;

    double _rho;

    size_t _num_evals;
    

public:
    SmoothDualDecompositionLBFGS(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            const Eigen::MatrixXi& edges,
                            const std::vector< std::vector<size_t> >& decomposition,
                            bool do_pruning=true);
    
    size_t run(double rho=1.0);

protected:
    static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
        return reinterpret_cast<SmoothDualDecompositionLBFGS*>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(const lbfgsfloatval_t *x,
			     lbfgsfloatval_t *g,
			     const int n,
			     const lbfgsfloatval_t step);

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        return reinterpret_cast<SmoothDualDecompositionLBFGS*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        printProgress(k, -fx, gnorm, step);
        return 0;
    }
};
#endif
