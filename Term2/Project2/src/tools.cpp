#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	unsigned long ctr = 0;
	VectorXd RMSE(estimations[0].size());
	RMSE.setZero();

	assert(estimations.size() != 0 && estimations.size() == ground_truth.size());
	for (ctr = 0; ctr < estimations.size(); ++ctr) {
		assert(estimations[ctr].size() == ground_truth[ctr].size() && ground_truth[ctr].size() == RMSE.size());
		RMSE = RMSE + VectorXd((estimations[ctr] - ground_truth[ctr]).array() * (estimations[ctr] - ground_truth[ctr]).array());
	}
	RMSE = (1.0/estimations.size()) * RMSE;
	RMSE = VectorXd(RMSE.array().sqrt());

	return RMSE;
}

MatrixXd Tools::GenerateSigmaPointMatrix(const VectorXd &x, const MatrixXd &P, double lambda) {
	MatrixXd Xsig = MatrixXd(x.size(), 2*x.size()+1);
	MatrixXd A = P.llt().matrixL();

	Xsig << x, x.replicate(1, A.cols()) + (A * sqrt(lambda + x.size())), x.replicate(1, A.cols()) - (A * sqrt(lambda + x.size()));

	return Xsig;
}

double Tools::NormalizeAngle(double angle_radians) {
	while (angle_radians > M_PI) {
		angle_radians -= 2*M_PI;
	}
	while (angle_radians < -M_PI) {
		angle_radians += 2*M_PI;
	}

	return angle_radians;
}
