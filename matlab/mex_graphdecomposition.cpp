#include <mex.h>
#include <Eigen/Core>
#include <GraphDecomposition.h>
#include "matlab_helpers.h"

// Input Arguments

#define EDGES_IN                prhs[0]
#define OPTIONS_IN              prhs[1]
#define NR_IN                   1 
#define NR_IN_OPT               1

// Output Arguments

#define DECOMPOSITION_OUT       plhs[0]
#define NR_OUT                  1
#define NR_OUT_OPT              0

/*class MexLPQPOptions {
public:
    double rho_start;
    double rho_end;
    size_t num_max_iter_dc;
    size_t num_max_iter_sdd;
    bool do_round;

    MexLPQPOptions() : 
        rho_start(1e-1),
        rho_end(1e5),
        num_max_iter_dc(30),
        num_max_iter_sdd(1000),
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
            else if (opt_name_str == "num_max_iter_dc") {
                num_max_iter_dc = static_cast<size_t>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "num_max_iter_sdd") {
                num_max_iter_sdd = static_cast<size_t>(mxGetScalar(opt_val));
            }
            else if (opt_name_str == "do_round") {
                do_round = static_cast<bool>(mxGetScalar(opt_val));
            }
            else {
                mexErrMsgTxt("Name of the option is invalid.\n");
                return (false);
            }
        }

        return true;
    }
};*/

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    MatlabCPPInitialize(false);

    // Check for proper number of arguments
    if ( (nrhs < NR_IN) || (nrhs > NR_IN + NR_IN_OPT) || \
            (nlhs < NR_OUT) || (nlhs > NR_OUT + NR_OUT_OPT) ) {
        mexErrMsgTxt("Wrong number of arguments.");
    }

    Eigen::MatrixXi edges;
    GetMatlabMatrix(EDGES_IN, edges);
    
    //// parse options
    //MexLPQPOptions options;
    //if (nrhs >= 2) {
    //    bool opts_parsed = options.parse(OPTIONS_IN);
    //    if (!opts_parsed) {
    //        MatlabCPPExit();
    //        return;
    //    }
    //}

    // compute decomposition
    GraphDecomposition graph_decomposition(edges);
    std::vector< std::vector<size_t> > decomposition;
    decomposition = graph_decomposition.decompose(GraphDecomposition::DECOMPOSITION_STACK);

    // return decomposition (cell array of vectors)
    mwSize dims[1];
    dims[0] = decomposition.size();
    DECOMPOSITION_OUT = mxCreateCellArray(1, dims);
    for (size_t d_idx=0; d_idx<decomposition.size(); d_idx++) {
        mxArray* decomp_array;
        decomp_array = mxCreateDoubleMatrix(decomposition[d_idx].size(),1,mxREAL);
        double* decomp_array_ptr = mxGetPr(decomp_array);
        for (size_t e_idx=0; e_idx<decomposition[d_idx].size(); e_idx++) {
            decomp_array_ptr[e_idx] = decomposition[d_idx][e_idx];
        }
        mxSetCell(DECOMPOSITION_OUT, d_idx, decomp_array);
    }

    MatlabCPPExit();
}
