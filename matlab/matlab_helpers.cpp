#include <matlab_helpers.h>
#include <matlab_out.h>
#include <mex.h>
#include <vector>
#include <Eigen/Core>

matlab_out mp;
std::streambuf* matlab_out::mp_backup = 0;

void MatlabCPPInitialize(bool verbose) {
    matlab_out::OutInitialize(&mp, verbose);
}

void MatlabCPPExit() {
    matlab_out::OutExit();
}

std::string GetMatlabString(const mxArray* m_str) {
	assert(mxIsChar(m_str));
	int char_count = static_cast<int>(mxGetNumberOfElements(m_str) + 1);
	char* buf = static_cast<char*>(mxCalloc(char_count, sizeof(char)));
	mxGetString(m_str, buf, char_count);

	std::string res(buf);
	mxFree(buf);

	return (res);
}

/*void GetMatlabVector(const mxArray* vin,
	std::vector<int>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		vout[n] = static_cast<int>(vec_p[n]);
}

void GetMatlabVector(const mxArray* vin,
	std::vector<bool>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		vout[n] = static_cast<bool>(vec_p[n]);
}

void GetMatlabVector(const mxArray* vin,
	std::vector<size_t>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n) {
		vout[n] = static_cast<size_t>(vec_p[n]);
    }
}

void GetMatlabVector(const mxArray* vin,
	std::vector<double>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		vout[n] = vec_p[n];
}*/

void GetMatlabVector(const mxArray* vin,
	Eigen::VectorXd& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
    if (vec_count > 0) {
	    vout.setZero(vec_count);
	    const double* vec_p = mxGetPr(vin);
	    for (unsigned int n = 0; n < vec_count; ++n)
	    	vout[n] = vec_p[n];
    }
}

void GetMatlabMatrix(const mxArray* min,
	Eigen::MatrixXd& mout) {
	if (mxIsDouble(min) == false) {
		mexErrMsgTxt("Passed matrix is no double matrix.\n");
		assert(0);
	}
    if (mxGetNumberOfDimensions(min) != 2) {
		mexErrMsgTxt("Passed array is not a matrix.\n");
		assert(0);
    }
	size_t M = mxGetM(min);
	size_t N = mxGetN(min);
	size_t vec_count = mxGetNumberOfElements(min);
    if (M > 0 && N > 0) {
	    mout.setZero(M, N);
	    const double* vec_p = mxGetPr(min);
	    for (unsigned int n = 0; n < vec_count; ++n)
	    	mout(n) = vec_p[n];
    }
}

void GetMatlabMatrix(const mxArray* min,
	Eigen::MatrixXi& mout) {
	if (mxIsDouble(min) == false) {
		mexErrMsgTxt("Passed matrix is no double matrix.\n");
		assert(0);
	}
    if (mxGetNumberOfDimensions(min) != 2) {
		mexErrMsgTxt("Passed array is not a matrix.\n");
		assert(0);
    }
	size_t M = mxGetM(min);
	size_t N = mxGetN(min);
	size_t vec_count = mxGetNumberOfElements(min);
    if (M > 0 && N > 0) {
	    mout.setZero(M, N);
	    const double* vec_p = mxGetPr(min);
	    for (unsigned int n = 0; n < vec_count; ++n)
	    	mout(n) = static_cast<int>(vec_p[n]);
    }
}

void GetMatlabPotential(const mxArray* pot_in,
        std::vector< Eigen::VectorXd >& pot_out)
{
    size_t num = mxGetNumberOfElements(pot_in);
    pot_out.resize(num);

    for (size_t idx=0; idx<num; idx++) {
        const mxArray* v = mxGetCell(pot_in, idx);
        GetMatlabVector(v, pot_out[idx]);
    }
}

void GetMatlabPotentialFromMatrix(const mxArray* pot_in,
        std::vector< Eigen::VectorXd >& pot_out)
{
    size_t num_vars = mxGetN(pot_in);
    size_t num_states = mxGetM(pot_in);
    pot_out.resize(num_vars);

	const double* ptr = mxGetPr(pot_in);
    for (size_t v_idx=0; v_idx<num_vars; v_idx++) {
        pot_out[v_idx].setZero(num_states);
        for (size_t k=0; k<num_states; k++) {
            pot_out[v_idx](k) = ptr[v_idx*num_states+k];
        }
    }
}
