#ifndef __DEFINED_LPQPSDD_H
#define __DEFINED_LPQPSDD_H

#include <LPQP.h>
#include <vector>
#include <Eigen/Core>
#include <SmoothDualDecomposition.h>

class LPQPSDD : public LPQP {

private:
    SmoothDualDecomposition* _sdd;

    Eigen::VectorXd _node_weight_concave;
    Eigen::VectorXd _edge_weight_kl;
    
    double computeKLDivergence();

public:
    enum Solver { LBFGS, FISTADESCENT };

    LPQPSDD(std::vector< Eigen::VectorXd >& theta_unary,
                std::vector< Eigen::VectorXd >& theta_pair,
                Eigen::MatrixXi& edges,
                const std::vector< std::vector<size_t> >& decomposition,
                Solver solver = FISTADESCENT
                );
    ~LPQPSDD();

    void run();
    
    double runInitialLP();

    void setEpsilonMP(double eps);
    void setMaximumNumberOfIterationsMP(size_t num_max_iter);
};

#endif
