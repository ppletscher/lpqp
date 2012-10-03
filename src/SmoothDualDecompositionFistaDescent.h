#ifndef __DEFINED_SMOOTHDUALDECOMPOSITIONFISTADESCENT_H
#define __DEFINED_SMOOTHDUALDECOMPOSITIONFISTADESCENT_H

#include <vector>
#include <Eigen/Core>
#include <TreeInference.h>
#include <SmoothDualDecomposition.h>

class SmoothDualDecompositionFistaDescent : public SmoothDualDecomposition {
private:
    // dual variables
    Eigen::VectorXd _lambda, _v, _y, _y_old, _u;
    
    // parameters for the line search for a gradient step
    double _lipschitz_constant_optimistic;
    double _lipschitz_inc_d;
    double _lipschitz_inc_u;
    
    bool stoppingCriteriaMet(size_t iter, double gnorm);

    double gradientStep(double rho,
            const Eigen::VectorXd& lambda,
            double& obj_lambda,
            Eigen::VectorXd& gradient,
            double& gradient_norm_squared,
            Eigen::VectorXd& y, double& omega);
    

public:
    SmoothDualDecompositionFistaDescent(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            const Eigen::MatrixXi& edges,
                            const std::vector< std::vector<size_t> >& decomposition,
                            bool do_pruning=true);
    
    size_t run(double rho=1.0);
};
#endif
