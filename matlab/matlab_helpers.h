#ifndef LPQP_MATLABHELPERS_H
#define LPQP_MATLABHELPERS_H

#include <mex.h>
#include <vector>
#include <Eigen/Core>

void MatlabCPPInitialize(bool verbose = true);
void MatlabCPPExit();


std::string GetMatlabString(const mxArray* m_str);

/*void GetMatlabVector(const mxArray* vin,
	std::vector<size_t>& vout);

void GetMatlabVector(const mxArray* vin,
	std::vector<bool>& vout);

void GetMatlabVector(const mxArray* vin,
	std::vector<double>& vout);*/

void GetMatlabVector(const mxArray* vin,
	Eigen::VectorXd& vout);

void GetMatlabMatrix(const mxArray* min,
	Eigen::MatrixXd& mout);

void GetMatlabMatrix(const mxArray* min,
	Eigen::MatrixXi& mout);

void GetMatlabPotential(const mxArray* pot_in,
        std::vector< Eigen::VectorXd >& pot_out);

void GetMatlabPotentialFromMatrix(const mxArray* pot_in,
        std::vector< Eigen::VectorXd >& pot_out);

#endif
