#ifndef __DEFINED_TREEINFERENCE_H
#define __DEFINED_TREEINFERENCE_H

#include <vector>
#include <Eigen/Core>

class TreeInference {

private:
    // unary and pairwise potentials. NOTE: this is for *maximization*, so
    // negative energy!
    std::vector< Eigen::VectorXd > _theta_unary; // dim: num_states*num_vertices
    std::vector< Eigen::VectorXd > _theta_pair; // dim: num_states^2*num_edges

    // edge list, matrix of dimension 2xnum_edges
    const Eigen::MatrixXi _edges;

    double _beta; // inverse temperature

    size_t _num_vertices;
    size_t _num_edges;
    std::vector< size_t > _num_states;

    // graph is stored as a list of neighbors for each variable
    std::vector< std::vector< std::pair<size_t,size_t> > > _neighbors;

    std::vector< Eigen::VectorXd > _mu_unary;
    std::vector< Eigen::VectorXd > _mu_pair;

    double _entropy;
    double _average_energy;
    double _logZ;
    double _free_energy;

    // first and second refers to the edges(0,e) (first) and edges(1,e) (second)
    std::vector< Eigen::VectorXd > _msg_to_first; // msg sent from second to first
    std::vector< Eigen::VectorXd > _msg_to_second; // msg sent from first to second

    // orders to pass messages upwards and downwards (2xnum_edges), first
    // entry stores edge_id, second entry stores the direction (0: same
    // direction as in _edges, 1: altered direction)
    Eigen::MatrixXi _order;

    void determineOrder();

    void passMessage(size_t e_idx, bool direction);
    Eigen::VectorXd collectIncomingAndUnary(size_t i, int exclude=-1);

    double logsumexp(const Eigen::VectorXd& x);
    void normalizeVector(Eigen::VectorXd& v);

    void setMarginalsUnary();
    void setMarginalsPair();

    void computeEntropy();
    void computeAverageEnergy();

    // only needed for the exhaustive computations
    double computeScore(const Eigen::VectorXi& label);
    void incrementLabel(Eigen::VectorXi& label);
    void addScoreToMarginals(double s, const Eigen::VectorXi& label);

    double log0(double x);

    void checkPotentials();

public:
    TreeInference(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges);

    void run(double beta=1.0);

    std::vector< Eigen::VectorXd >& getUnaryMarginals();

    std::vector< Eigen::VectorXd >& getPairwiseMarginals();

    double getLogPartitionSum();

    double getFreeEnergy();

    void setThetaUnary(size_t i, Eigen::VectorXd& theta);
    void setThetaPair(size_t e, Eigen::VectorXd& theta);

    void runExhaustive(double beta);
};

#endif
