#include <mex.h>
#include <Eigen/Core>
#include <TRWS.h>
#include "matlab_helpers.h"

// Input Arguments

#define THETA_UNARY_IN          prhs[0]
#define THETA_PAIR_IN           prhs[1]
#define EDGES_IN                prhs[2]
#define OPTIONS_IN              prhs[3]
#define NR_IN                   3 
#define NR_IN_OPT               1

// Output Arguments

#define MARGINALS_UNARY_OUT     plhs[0]
#define NR_OUT                  1
#define NR_OUT_OPT              0

class MexTRWSOptions {
public:
    size_t num_max_iter;

    MexTRWSOptions() : 
        num_max_iter(500)
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

    // parse options
    MexTRWSOptions options;
    if (nrhs >= 4) {
        bool opts_parsed = options.parse(OPTIONS_IN);
        if (!opts_parsed) {
            MatlabCPPExit();
            return;
        }
    }

    TRWS trws(theta_unary, theta_pair, edges);

    // set options
    trws.setMaximumNumberOfIterations(options.num_max_iter);

    trws.run();
    
    // return marginals (if input was provided as a matrix, also return
    // marginals in a matrix)
    std::vector< Eigen::VectorXd >& mu_unary = trws.getUnaryMarginals();
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

    MatlabCPPExit();
}
