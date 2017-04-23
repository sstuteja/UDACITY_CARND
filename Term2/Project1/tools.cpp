#include <iostream>
#include "tools.h"
#include <cassert>
#include <cmath>

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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);

	float vy = x_state(3);

	MatrixXd JACOBIAN(3, 4);

	float rho = sqrt(px*px + py*py);
	JACOBIAN << px/rho,	py/rho, 0.0, 0.0,
				-py/(rho*rho), px/(rho*rho), 0.0, 0.0,
				py*(vx*py - vy*px)/(rho*rho*rho), px*(vy*px - vx*py)/(rho*rho*rho), px/rho, py/rho;

	return JACOBIAN;
}
