#ifndef __DEFINED_LPQP_H
#define __DEFINED_LPQP_H

#include <vector>
#include <Eigen/Core>

class LPQP {

protected:
    // unary and pairwise potentials. NOTE: this is for *minimization*.
    std::vector< Eigen::VectorXd > _theta_unary; // dim: num_states*num_vertices
    std::vector< Eigen::VectorXd > _theta_pair; // dim: num_states^2*num_edges

    // edge list, matrix of dimension 2xnum_edges
    const Eigen::MatrixXi _edges;

    // graph is stored as a list of neighbors for each variable
    std::vector< std::vector< std::pair<size_t,size_t> > > _neighbors;

    double _rho_start;
    double _rho_end;

    double _eps_entropy;
    double _eps_dkl;
    double _eps_obj;

    double _eps_mp;

    size_t _num_max_iter_dc;
    size_t _num_max_iter_mp;

    double _rho_schedule_constant;

    bool _initial_lp_active;
    double _initial_lp_improvement_ratio;
    double _initial_lp_rho_start;
    bool _initial_rho_similar_values;
    double _initial_rho_factor_kl_smaller;

    bool _skip_if_increase; // go to next CCCP step if in MP objective increases?

    size_t _num_vertices;
    size_t _num_edges;
    std::vector< size_t > _num_states;
    
    // the unary potentials will change over time, _theta_unary_curr always
    // stores the current potentials (hence the message-passing algorithm
    // works with these potentials)
    std::vector< Eigen::VectorXd > _theta_unary_curr;

    std::vector< Eigen::VectorXd > _mu_unary;
    std::vector< Eigen::VectorXd > _mu_pair;

    std::vector<double> _history_obj;
    std::vector<double> _history_obj_qp;
    std::vector<double> _history_obj_lp;
    std::vector<double> _history_obj_decoded;
    std::vector<size_t> _history_iter;
    std::vector<double> _history_rho;
    std::vector< Eigen::VectorXi > _history_decoded;


    double computeObjective(double beta);
    double computeUnaryEntropy();
    double computeObjectiveDecoded();
    virtual double computeKLDivergence() = 0;

    Eigen::VectorXi getDecodedAssignment();

    void printProgressHeader();

    void printProgress(double beta, size_t dc_iter, size_t mp_iter,
                        double val, double val_qp, double val_lp, double val_decoded);

    double determineInitialRho();

    double log0(double x);

public:
    LPQP(std::vector< Eigen::VectorXd >& theta_unary,
                            std::vector< Eigen::VectorXd >& theta_pair,
                            Eigen::MatrixXi& edges);
    virtual ~LPQP();

    virtual void run() = 0;

    std::vector< Eigen::VectorXd >& getUnaryMarginals();

    std::vector< Eigen::VectorXd >& getPairwiseMarginals();

    void setRhoStart(double rho_start);
    void setRhoEnd(double rho_end);

    void setEpsilonEntropy(double eps_entropy);
    void setEpsilonKullbackLeibler(double eps_dkl);
    void setEpsilonObjective(double eps_obj);

    void setMaximumNumberOfIterationsDC(size_t num_max_iter);
    
    virtual void setEpsilonMP(double eps);
    virtual void setMaximumNumberOfIterationsMP(size_t num_max_iter);
    
    void setRhoScheduleConstant(double c);
    void setInitialLPActive(bool initial_lp_active);
    void setInitialLPImprovmentRatio(double initial_lp_improvement_ratio);
    void setInitialLPRhoStart(double initial_lp_rho_start);
    void setInitialRhoSimilarValues(bool initial_rho_similar_values);
    void setInitialRhoFactorKLSmaller(double initial_rho_factor_kl_smaller);
    void setSkipIfIncrease(bool skip_if_increase);

    void roundSolution();

    double computeQPValue();
    double computeLPValue();

    virtual double runInitialLP() = 0;
    
    std::vector<double>& getHistoryObjective();
    std::vector<double>& getHistoryObjectiveQP();
    std::vector<double>& getHistoryObjectiveLP();
    std::vector<double>& getHistoryObjectiveDecoded();
    std::vector<size_t>& getHistoryIteration();
    std::vector<double>& getHistoryRho();
};

#endif
