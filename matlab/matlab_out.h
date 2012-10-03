#ifndef SDLM_MATLABOUT_H
#define SDLM_MATLABOUT_H

#include <streambuf>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <mex.h>

class matlab_out : public std::streambuf {
public:
	explicit matlab_out(bool verbose = true)
		: verbose(verbose) {
	}

	virtual int overflow(int cn = -1) {
		if (cn != -1 && verbose) {
			mexPrintf("%c", cn);
			mexEvalString("drawnow");
		}
		return (1);
	}
	virtual std::streamsize xsputn(const char* str, std::streamsize str_n) {
		if (verbose) {
			mexPrintf("%.*s", str_n, str);
			mexEvalString("drawnow");
		}
		return (str_n);
	}

	static void OutInitialize(matlab_out* mp, bool verbose) {
		mp->verbose = verbose;
		mp_backup = std::cout.rdbuf(mp);
	}
	static void OutExit() {
		std::cout.rdbuf(mp_backup);
	}

private:
	bool verbose;
	static std::streambuf* mp_backup;
};

#endif

