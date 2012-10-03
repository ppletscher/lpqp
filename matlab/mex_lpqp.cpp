#include <mex.h>
#include <Eigen/Core>
#include <LPQP.h>
#include <LPQPSDD.h>
#include <LPQPNPBP.h>
#include "matlab_helpers.h"

// Input Arguments

#define THETA_UNARY_IN          prhs[0]
#define THETA_PAIR_IN           prhs[1]
#define EDGES_IN                prhs[2]
#define OPTIONS_IN              prhs[3]
#define DECOMPOSITION_IN        prhs[4]
#define NR_IN                   3 
#define NR_IN_OPT               2

// Output Arguments

#define MARGINALS_UNARY_OUT     plhs[0]
#define HISTORY_OUT             plhs[1]
#define NR_OUT                  1
#define NR_OUT_OPT              1

//enum SolverType { SOLVER_FISTADESCENT, SOLVER_LBFGS };

class MexLPQPOptions {
public:
    double rho_start;
    double rho_end;
    double eps_entropy;
    double eps_dkl;
    double eps_obj;
    double eps_mp;
    size_t num_max_iter_dc;
    size_t num_max_iter_mp;
    double rho_schedule_constant;
    bool initial_lp_active;
    double initial_lp_improvement_ratio;
    double initial_lp_rho_start;
    bool initial_rho_similar_values;
    double initial_rho_factor_kl_smaller;
    bool skip_if_increase;
    LPQPSDD::Solver solver_sdd;
    bool do_round;

    MexLPQPOptions() : 
        rho_start(1e-1),
        rho_end(1e10),
        eps_entropy(1e-3),
        eps_dkl(1e-6),
        eps_obj(1e-3),
        eps_mp(1e-4),
        num_max_iter_dc(60),
        num_max_iter_mp(300),
        rho_schedule_constant(1.5),
        initial_lp_active(false),
        initial_lp_improvement_ratio(1e-2),
        initial_lp_rho_start(25),
        initial_rho_similar_values(false),
        initial_rho_factor_kl_smaller(10),
        skip_if_increase(true),
        solver_sdd(LPQPSDD::FISTADESCENT),
        do_round(true)
        {}

    bool parse(const mxArray* opts_in)
    {
        if (!mxIsStruct(opts_in)) {
            mexErrMsgTxt("Options parameter must be a structure.\n");
            return (false);
        }
        int num_fields = mxGetNumberOfFields(opts_in);
        
        for (int fn=0; fn<num_fields; fn++) {
            const char* opt_name = mxGetFieldNameByNumber(opts_in, fn);
            std::string opt_name_str = opt_name;
            mxArray *opt_val = mxGetFieldByNumber(opts_in, 0, fn);
            if (opt_name_str == "rho_start") {
                rho_start = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "rho_end") {
                rho_end = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "eps_dkl") {
                eps_dkl = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "eps_obj") {
                eps_obj = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "eps_mp") {
                eps_mp = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "eps_entropy") {
                eps_entropy = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "num_max_iter_dc") {
                num_max_iter_dc = static_cast<size_t>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "num_max_iter_mp") {
                num_max_iter_mp = static_cast<size_t>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "rho_schedule_constant") {
                rho_schedule_constant = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "do_round") {
                do_round = static_cast<bool>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "initial_lp_active") {
                initial_lp_active = static_cast<bool>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "initial_lp_improvement_ratio") {
                initial_lp_improvement_ratio = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "initial_lp_rho_start") {
                initial_lp_rho_start = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "initial_rho_similar_values") {
                initial_rho_similar_values = static_cast<bool>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "initial_rho_factor_kl_smaller") {
                initial_rho_factor_kl_smaller = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "skip_if_increase") {
                skip_if_increase = static_cast<bool>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "solver_sdd") {
                std::string solver_name = GetMatlabString(opt_val);
                if (solver_name == "fistadescent") {
                    solver_sdd = LPQPSDD::FISTADESCENT;
                }
                else if (solver_name == "lbfgs") {
                    solver_sdd = LPQPSDD::LBFGS;
                }
                else {
                    mexErrMsgTxt("No solver with this name.\n");
                    return false;
                }
            }
            else {
                mexErrMsgTxt("Name of the option is invalid.\n");
                return (false);
            }
        }

        return true;
    }
};

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    MatlabCPPInitialize(false);

    // Check for proper number of arguments
    if ( (nrhs < NR_IN) || (nrhs > NR_IN + NR_IN_OPT) || \
            (nlhs < NR_OUT) || (nlhs > NR_OUT + NR_OUT_OPT) ) {
        mexErrMsgTxt("Wrong number of arguments.");
    }

    // fetch the input (we support matrices and cell arrays)
    std::vector< Eigen::VectorXd > theta_unary;
    std::vector< Eigen::VectorXd > theta_pair;

    if (!mxIsCell(THETA_UNARY_IN)) {
        GetMatlabPotentialFromMatrix(THETA_UNARY_IN, theta_unary);
        GetMatlabPotentialFromMatrix(THETA_PAIR_IN, theta_pair);
    }
    else {
        GetMatlabPotential(THETA_UNARY_IN, theta_unary);
        GetMatlabPotential(THETA_PAIR_IN, theta_pair);
    }
    
    Eigen::MatrixXi edges;
    GetMatlabMatrix(EDGES_IN, edges);

    // decomposition: cell array of edge index vectors
    std::vector< std::vector<size_t> > decomposition;
    if (nrhs >= 5) {
        size_t num = mxGetNumberOfElements(DECOMPOSITION_IN);
        decomposition.resize(num);
        for (size_t d_idx=0; d_idx<num; d_idx++) {
            const mxArray* v = mxGetCell(DECOMPOSITION_IN, d_idx);
            size_t num_edges = mxGetM(v);
            decomposition[d_idx].resize(num_edges);
	        const double* ptr = mxGetPr(v);
            for (size_t e_idx=0; e_idx<num_edges; e_idx++) {
                decomposition[d_idx][e_idx] = static_cast<size_t>(ptr[e_idx]);
            }
        }
    }
    
    // parse options
    MexLPQPOptions options;
    if (nrhs >= 4) {
        bool opts_parsed = options.parse(OPTIONS_IN);
        if (!opts_parsed) {
            MatlabCPPExit();
            return;
        }
    }

    LPQP* lpqp;
    if (nrhs < 5) {
        lpqp = new LPQPNPBP(theta_unary, theta_pair, edges);
    }
    else {
        lpqp = new LPQPSDD(theta_unary, theta_pair, edges, decomposition, options.solver_sdd);
    }

    lpqp->setRhoStart(options.rho_start);
    lpqp->setRhoEnd(options.rho_end);
    lpqp->setEpsilonEntropy(options.eps_entropy);
    lpqp->setEpsilonKullbackLeibler(options.eps_dkl);
    lpqp->setEpsilonObjective(options.eps_obj);
    lpqp->setEpsilonMP(options.eps_mp);
    lpqp->setMaximumNumberOfIterationsDC(options.num_max_iter_dc);
    lpqp->setMaximumNumberOfIterationsMP(options.num_max_iter_mp);
    lpqp->setRhoScheduleConstant(options.rho_schedule_constant);
    lpqp->setInitialLPActive(options.initial_lp_active);
    lpqp->setInitialLPImprovmentRatio(options.initial_lp_improvement_ratio);
    lpqp->setInitialLPRhoStart(options.initial_lp_rho_start);
    lpqp->setInitialRhoSimilarValues(options.initial_rho_similar_values);
    lpqp->setInitialRhoFactorKLSmaller(options.initial_rho_factor_kl_smaller);
    lpqp->setSkipIfIncrease(options.skip_if_increase);

    lpqp->run();
    
    if (options.do_round) {
        double curr_qp_obj = lpqp->computeQPValue();

        lpqp->roundSolution();

        double qp_obj_after_rounding = lpqp->computeQPValue();
        printf("QP objective before rounding: %f after rounding: %f\n",curr_qp_obj, qp_obj_after_rounding);
    }

    // return marginals (if input was provided as a matrix, also return
    // marginals in a matrix)
    std::vector< Eigen::VectorXd >& mu_unary = lpqp->getUnaryMarginals();
    if (!mxIsCell(THETA_UNARY_IN)) {
        size_t num_states = mxGetM(THETA_UNARY_IN);
        size_t num_vars = mxGetN(THETA_UNARY_IN);
        MARGINALS_UNARY_OUT = mxCreateNumericMatrix(
                                num_states, 
                                num_vars,
                                mxDOUBLE_CLASS, mxREAL);
        double* mu_res_p = mxGetPr(MARGINALS_UNARY_OUT);
        for (size_t v_idx=0; v_idx<mu_unary.size(); v_idx++) {
            for (size_t idx=0; idx<mu_unary[v_idx].size(); idx++) {
                mu_res_p[v_idx*num_states+idx] = mu_unary[v_idx](idx);
            }
        }
    }
    else {
        mwSize dim_0 = static_cast<mwSize>(mu_unary.size());
        MARGINALS_UNARY_OUT = mxCreateCellArray(1, &dim_0);
        for (size_t v_idx=0; v_idx<mu_unary.size(); v_idx++) {
            mxArray* m = mxCreateNumericMatrix(
                            static_cast<int>(mu_unary[v_idx].size()),
                            1,
                            mxDOUBLE_CLASS, mxREAL);
            double* mu_res_p = mxGetPr(m);
            for (size_t idx=0; idx<mu_unary[v_idx].size(); idx++) {
                mu_res_p[idx] = mu_unary[v_idx](idx);
            }

            mxSetCell(MARGINALS_UNARY_OUT, v_idx, m);
        }
    }
    
    // return history
    if (nlhs > 1) {
        std::vector<double>& history_obj = lpqp->getHistoryObjective();
        std::vector<double>& history_obj_qp = lpqp->getHistoryObjectiveQP();
        std::vector<double>& history_obj_lp = lpqp->getHistoryObjectiveLP();
        std::vector<double>& history_obj_decoded = lpqp->getHistoryObjectiveDecoded();
        std::vector<size_t>& history_iteration = lpqp->getHistoryIteration();
        std::vector<double>& history_rho = lpqp->getHistoryRho();
        //std::vector< Eigen::VectorXi > history_decoded = lpqp.getHistoryDecoded();

        //const char *field_names[] = {"obj", "obj_qp", "obj_lp", "obj_decoded", "iteration", "beta", "decoded"};
        const char *field_names[] = {"obj", "obj_qp", "obj_lp", "obj_decoded", "iteration", "rho"};
        mwSize dims[2];
        dims[0] = 1;
        dims[1] = history_obj.size();
        HISTORY_OUT = mxCreateStructArray(2, dims, sizeof(field_names)/sizeof(*field_names), field_names);

        int field_obj, field_obj_qp, field_obj_lp, field_obj_decoded, field_iteration, field_rho;
        //int field_decoded;
        field_obj = mxGetFieldNumber(HISTORY_OUT, "obj");
        field_obj_qp = mxGetFieldNumber(HISTORY_OUT, "obj_qp");
        field_obj_lp = mxGetFieldNumber(HISTORY_OUT, "obj_lp");
        field_obj_decoded = mxGetFieldNumber(HISTORY_OUT, "obj_decoded");
        field_iteration = mxGetFieldNumber(HISTORY_OUT, "iteration");
        field_rho = mxGetFieldNumber(HISTORY_OUT, "rho");
        //field_decoded = mxGetFieldNumber(HISTORY_OUT, "decoded");

        for (size_t i=0; i<history_obj.size(); i++) {
            mxArray *field_value;

            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = history_obj[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_obj, field_value);
            
            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = history_obj_qp[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_obj_qp, field_value);
            
            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = history_obj_lp[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_obj_lp, field_value);
            
            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = history_obj_decoded[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_obj_decoded, field_value);
            
            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = static_cast<double>(history_iteration[i]);
            mxSetFieldByNumber(HISTORY_OUT, i, field_iteration, field_value);
            
            field_value = mxCreateDoubleMatrix(1,1,mxREAL);
            *mxGetPr(field_value) = history_rho[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_rho, field_value);
            
            //if (options.with_decoded_history) {
            //    field_value = mxCreateDoubleMatrix(history_decoded[i].size(),1,mxREAL);
            //    double* decoded_res_p = mxGetPr(field_value);
            //    for (size_t idx=0; idx<history_decoded[i].size(); idx++) {
            //        decoded_res_p[idx] = static_cast<double>(history_decoded[i][idx]);
            //    }
            //    mxSetFieldByNumber(HISTORY_OUT, i, field_decoded, field_value);
            //}
        }
    }

    delete lpqp;

    MatlabCPPExit();
}
