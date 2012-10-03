#ifndef __DEFINED_LPQPNPBP_H
#define __DEFINED_LPQPNPBP_H

#include <LPQP.h>
#include <vector>
#include <Eigen/Core>

class LPQPNPBP : public LPQP {

private:
    
    // first and second refers to the edges(0,e) (first) and edges(1,e) (second)
    std::vector< Eigen::VectorXd > _msg_to_first; // msg sent from second to first
    std::vector< Eigen::VectorXd > _msg_to_second; // msg sent from first to second

    size_t passMessages(double beta);

    double sendMessages(size_t v, double beta);

    double logsumexp(const Eigen::VectorXd& x);

    double updateMarginalsUnary(double beta);
    void updateMarginalsPair(double beta);

    Eigen::VectorXd collectIncomingAndUnary(size_t i, double beta);

    void normalizeVector(Eigen::VectorXd& v);

    void checkVectorNotNan(Eigen::VectorXd& m, size_t idx);
    
    double computeKLDivergence();

public:
    LPQPNPBP(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges);

    void run();
    
    double runInitialLP();
};

#endif
