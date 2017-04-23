#include "kalman_filter.h"
#include "tools.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &Hj_in, MatrixXd &R_lidar_in, MatrixXd &R_radar_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  Hj_ = Hj_in;
  R_lidar_ = R_lidar_in;
  R_radar_ = R_radar_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */

	VectorXd y = VectorXd(z.size());
	MatrixXd S = MatrixXd(R_lidar_.rows(), R_lidar_.cols());
	MatrixXd K = MatrixXd(x_.size(), y.size());

	y = z - H_ * x_;
	S = H_ * P_ * H_.transpose() + R_lidar_;
	K = P_ * H_.transpose() * S.inverse();
	x_ = x_ + K*y;
	P_ = (MatrixXd::Identity(K.rows(), H_.cols()) - K*H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

	VectorXd y = VectorXd(z.size());
	MatrixXd S = MatrixXd(R_radar_.rows(), R_radar_.cols());
	MatrixXd K = MatrixXd(x_.size(), y.size());

	VectorXd HofX = VectorXd(z.size());

	Tools toolobj;

	Hj_ = toolobj.CalculateJacobian(x_);

	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);
	float rho = sqrt(px*px + py*py);
	if (rho < 1e-8) {
		//Skip
		return;
	}

	float phi = atan2(py, px);
	float rhodot = (px*vx + py*vy)/rho;

	//Making sure that phi lies between -M_PI and +M_PI
	if (!(phi >= -M_PI && phi <= M_PI)) {
		phi = phi + 2*M_PI;
		if (!(phi >= -M_PI && phi <= M_PI)) {
			phi = phi - 4*M_PI;
		}
	}
	assert(phi >= -M_PI && phi <= M_PI);

	HofX << rho,
			phi,
			rhodot;

	y = z - HofX;
	S = Hj_ * P_ * Hj_.transpose() + R_radar_;
	K = P_ * Hj_.transpose() * S.inverse();
	x_ = x_ + K*y;
	P_ = (MatrixXd::Identity(K.rows(), Hj_.cols()) - K*Hj_) * P_;
}
