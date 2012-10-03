#include <mex.h>
#include <Eigen/Core>
#include <TreeInference.h>
#include "matlab_helpers.h"

// Input Arguments

#define THETA_UNARY_IN          prhs[0]
#define THETA_PAIR_IN           prhs[1]
#define EDGES_IN                prhs[2]
#define BETA_IN                 prhs[3]
//#define OPTIONS_IN              prhs[3]
#define NR_IN                   4
#define NR_IN_OPT               0

// Output Arguments

#define LOGZ_OUT                plhs[0]
#define MARGINALS_UNARY_OUT     plhs[1]
#define MARGINALS_PAIR_OUT      plhs[2]
#define NR_OUT                  1
#define NR_OUT_OPT              2

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

    double beta = mxGetScalar(BETA_IN);
    
    TreeInference inf = TreeInference(theta_unary, theta_pair, edges);
    inf.run(beta);
            
    LOGZ_OUT = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double* logz_p = mxGetPr(LOGZ_OUT);
    logz_p[0] = inf.getLogPartitionSum();
    
    // return marginals (if input was provided as a matrix, also return
    // marginals in a matrix)
    if (nlhs > 1) {
        std::vector< Eigen::VectorXd >& mu_unary = inf.getUnaryMarginals();
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
    }

    // return pairwise marginals
    if (nlhs > 2) {
        std::vector< Eigen::VectorXd >& mu_pair = inf.getPairwiseMarginals();
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
            mxAssert(0, "not implemented yet!");
            // TODO: not implemented yet, see above!
            // see unary case above!
        }
    }

    MatlabCPPExit();
}
