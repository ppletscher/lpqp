#include <mex.h>
#include <Eigen/Core>
#include <SmoothDualDecompositionFistaDescent.h>
#include <SmoothDualDecompositionLBFGS.h>
#include "matlab_helpers.h"

// Input Arguments

#define THETA_UNARY_IN          prhs[0]
#define THETA_PAIR_IN           prhs[1]
#define EDGES_IN                prhs[2]
#define DECOMPOSITION_IN        prhs[3]
#define OPTIONS_IN              prhs[4]
#define NR_IN                   4 
#define NR_IN_OPT               1

// Output Arguments

#define MARGINALS_UNARY_OUT     plhs[0]
#define MARGINALS_PAIR_OUT      plhs[1]
//#define HISTORY_OUT             plhs[1]
#define NR_OUT                  1
#define NR_OUT_OPT              1

enum SolverType { SOLVER_FISTADESCENT, SOLVER_LBFGS };

class MexSmoothDDOptions {
public:
    double rho;
    size_t num_max_iter;
    double eps_gnorm;
    SolverType solver;

    MexSmoothDDOptions() : 
        rho(1e-3),
        num_max_iter(200),
        eps_gnorm(1e-9),
        solver(SOLVER_FISTADESCENT)
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
            if (opt_name_str == "num_max_iter") {
                num_max_iter = static_cast<size_t>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "rho") {
                rho = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "eps_gnorm") {
                eps_gnorm = mxGetScalar(opt_val);
            }
            else if (opt_name_str == "solver") {
                std::string solver_name = GetMatlabString(opt_val);
                if (solver_name == "fistadescent") {
                    solver = SOLVER_FISTADESCENT;
                }
                else if (solver_name == "lbfgs") {
                    solver = SOLVER_LBFGS;
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
    
    // parse options
    MexSmoothDDOptions options;
    if (nrhs >= 5) {
        bool opts_parsed = options.parse(OPTIONS_IN);
        if (!opts_parsed) {
            MatlabCPPExit();
            return;
        }
    }
    
    // set algorithm & parameters according to options
    SmoothDualDecomposition* sdd;
    /*if (options.solver == SOLVER_FISTADESCENT) {
        sdd = new SmoothDualDecompositionFistaDescent(theta_unary, theta_pair, edges, decomposition);
    }
    if (options.solver == SOLVER_FISTA) {
        sdd = new SmoothDualDecompositionFista(theta_unary, theta_pair, edges, decomposition);
    }
    else if (options.solver == SOLVER_GRADIENTDESCENT) {
        sdd = new SmoothDualDecompositionGradientDescent(theta_unary, theta_pair, edges, decomposition);
    }
    else if (options.solver == SOLVER_NESTEROV) {
        sdd = new SmoothDualDecompositionNesterov(theta_unary, theta_pair, edges, decomposition);
    }
    else {
        mxAssert(0, "Solver not found. Should not happen!");
    }*/
    // TODO
    sdd = new SmoothDualDecompositionFistaDescent(theta_unary, theta_pair, edges, decomposition);
    //sdd = new SmoothDualDecompositionLBFGS(theta_unary, theta_pair, edges, decomposition);
    sdd->setMaximumNumberOfIterations(options.num_max_iter);
    sdd->setEpsilonGradientNorm(options.eps_gnorm);

    // run the computations
    sdd->run(options.rho);

    // return marginals (if input was provided as a matrix, also return
    // marginals in a matrix)
    std::vector< Eigen::VectorXd >& mu_unary = sdd->getUnaryMarginals();
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
    /*if (nlhs > 1) {
        std::vector<double> history_obj = lpqp.getHistoryObjective();
        std::vector<double> history_obj_qp = lpqp.getHistoryObjectiveQP();
        std::vector<double> history_obj_lp = lpqp.getHistoryObjectiveLP();
        std::vector<double> history_obj_decoded = lpqp.getHistoryObjectiveDecoded();
        std::vector<size_t> history_iteration = lpqp.getHistoryIteration();
        std::vector<double> history_beta = lpqp.getHistoryBeta();
        std::vector< Eigen::VectorXi > history_decoded = lpqp.getHistoryDecoded();

        const char *field_names[] = {"obj", "obj_qp", "obj_lp", "obj_decoded", "iteration", "beta", "decoded"};
        mwSize dims[2];
        dims[0] = 1;
        dims[1] = history_obj.size();
        HISTORY_OUT = mxCreateStructArray(2, dims, sizeof(field_names)/sizeof(*field_names), field_names);

        int field_obj, field_obj_qp, field_obj_lp, field_obj_decoded, field_iteration, field_beta, field_decoded;
        field_obj = mxGetFieldNumber(HISTORY_OUT, "obj");
        field_obj_qp = mxGetFieldNumber(HISTORY_OUT, "obj_qp");
        field_obj_lp = mxGetFieldNumber(HISTORY_OUT, "obj_lp");
        field_obj_decoded = mxGetFieldNumber(HISTORY_OUT, "obj_decoded");
        field_iteration = mxGetFieldNumber(HISTORY_OUT, "iteration");
        field_beta = mxGetFieldNumber(HISTORY_OUT, "beta");
        field_decoded = mxGetFieldNumber(HISTORY_OUT, "decoded");

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
            *mxGetPr(field_value) = history_beta[i];
            mxSetFieldByNumber(HISTORY_OUT, i, field_beta, field_value);
            
            if (options.with_decoded_history) {
                field_value = mxCreateDoubleMatrix(history_decoded[i].size(),1,mxREAL);
                double* decoded_res_p = mxGetPr(field_value);
                for (size_t idx=0; idx<history_decoded[i].size(); idx++) {
                    decoded_res_p[idx] = static_cast<double>(history_decoded[i][idx]);
                }
                mxSetFieldByNumber(HISTORY_OUT, i, field_decoded, field_value);
            }
        }
    }*/
    
    // return pairwise marginals
    if (nlhs > 1) {
        std::vector< Eigen::VectorXd >& mu_pair = sdd->getPairwiseMarginals();
        if (!mxIsCell(THETA_UNARY_IN)) {
            size_t num_states = mxGetM(THETA_PAIR_IN);
            size_t num_edges = mxGetN(THETA_PAIR_IN);
            MARGINALS_PAIR_OUT = mxCreateNumericMatrix(
                                    num_states, 
                                    num_edges,
                                    mxDOUBLE_CLASS, mxREAL);
            double* mu_res_p = mxGetPr(MARGINALS_PAIR_OUT);
            for (size_t e_idx=0; e_idx<mu_pair.size(); e_idx++) {
                for (size_t idx=0; idx<mu_pair[e_idx].size(); idx++) {
                    mu_res_p[e_idx*num_states+idx] = mu_pair[e_idx](idx);
                }
            }
        }
        else {
            mwSize dim_0 = static_cast<mwSize>(mu_pair.size());
            MARGINALS_PAIR_OUT = mxCreateCellArray(1, &dim_0);
            for (size_t e_idx=0; e_idx<mu_pair.size(); e_idx++) {
                mxArray* m = mxCreateNumericMatrix(
                                static_cast<int>(mu_pair[e_idx].size()),
                                1,
                                mxDOUBLE_CLASS, mxREAL);
                double* mu_res_p = mxGetPr(m);
                for (size_t idx=0; idx<mu_pair[e_idx].size(); idx++) {
                    mu_res_p[idx] = mu_pair[e_idx](idx);
                }

                mxSetCell(MARGINALS_PAIR_OUT, e_idx, m);
            }
        }
    }

    delete sdd;

    MatlabCPPExit();
}
