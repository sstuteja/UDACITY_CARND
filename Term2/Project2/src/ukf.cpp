#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  //Set initialization flag to false
  is_initialized_ = false;

  //Example predicted state covariance matrix
  P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
		0, std_laspy_*std_laspy_, 0, 0, 0,
		0, 0, std_radrd_*std_radrd_, 0, 0,
		0, 0, 0, std_radphi_*std_radphi_, 0,
		0, 0, 0, 0, 0.05;

  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;

  //Number of state dimensions
  n_x_ = 5;

  //Number of augmented state dimensions
  n_aug_ = 7;

  //Lambda parameter
  lambda_ = 3 - n_aug_; //Based on rule

  time_us_ = 0;

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  //Set up weights
  	weights_ = VectorXd(Xsig_pred_.cols());
  	unsigned ctr = 0;
  	weights_(0) = lambda_ / (lambda_ + n_aug_);
  	for (ctr = 1; ctr < Xsig_pred_.cols(); ++ctr) {
  		weights_(ctr) = 1/(2*(lambda_ + n_aug_));
  	}
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	double rho, phi, rhodot;
	if (!is_initialized_) {
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			rho = meas_package.raw_measurements_(0);
			phi = meas_package.raw_measurements_(1);
			rhodot = meas_package.raw_measurements_(2);
			x_ << rho * cos(phi),
				  rho * sin(phi),
				  rhodot, //Assuming initially that phi = psi (i.e. straight line motion)
				  phi, //Assuming initially that phi = psi, i.e. straight line motion
				  0; //Straight line motion implies psidot = 0
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_ << meas_package.raw_measurements_(0),
				  meas_package.raw_measurements_(1),
				  0,
				  0,
				  0;
		}

		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
		return;
	}

	double dt = (meas_package.timestamp_ - time_us_)/1000000.0; //dt expressed in seconds
	time_us_ = meas_package.timestamp_;

	Prediction(dt);
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	unsigned ctr;

	//Create process noise covariance matrix Q
	MatrixXd Q = MatrixXd(2, 2);
	Q << std_a_ * std_a_, 0,
		 0, std_yawdd_ * std_yawdd_;

	//Generate augmented state matrix
	VectorXd x_aug = VectorXd(7);
	x_aug << x_,
			 0,
			 0;

	//Create augmented covariance matrix
	MatrixXd P_aug = MatrixXd(7, 7);
	P_aug << P_, MatrixXd::Zero(P_.rows(), Q.cols()),
	         MatrixXd::Zero(Q.rows(), P_.cols()), Q;

	//Generate sigma points in Xsig_aug
	Tools toolobj;
	MatrixXd Xsig_aug = toolobj.GenerateSigmaPointMatrix(x_aug, P_aug, lambda_);

	//Predict sigma points in Xsig_pred_
	double v, psi, psidot, nua, nupsidotdot;
	VectorXd mat1 = VectorXd(n_x_);
	VectorXd mat2 = VectorXd(n_x_);
	unsigned colctr = 0;
	for (colctr = 0; colctr < Xsig_pred_.cols(); ++colctr) {
		v = Xsig_aug(2, colctr);
		psi = Xsig_aug(3, colctr);
		psidot = Xsig_aug(4, colctr);
		nua = Xsig_aug(5, colctr);
		nupsidotdot = Xsig_aug(6, colctr);

		if (fabs(psidot) <= 1.0e-6) {
			mat1 << v * delta_t * cos(psi),
			   	    v * delta_t * sin(psi),
					0.0,
					psidot * delta_t,
					0.0;
		}
		else {
			mat1 << (v/psidot) * (sin(psi + psidot * delta_t) - sin(psi)),
			 	    (v/psidot) * (cos(psi) - cos(psi + psidot * delta_t)),
					0.0,
					psidot * delta_t,
					0.0;
		}

		mat2 << 0.5 * delta_t * delta_t * cos(psi) * nua,
			    0.5 * delta_t * delta_t * sin(psi) * nua,
				delta_t * nua,
				0.5 * delta_t * delta_t * nupsidotdot,
				delta_t * nupsidotdot;

		Xsig_pred_.col(colctr) = Xsig_aug.col(colctr).head(n_x_) + mat1 + mat2;
	}

	//Predict mean
	x_ = VectorXd::Zero(n_x_);
	for (unsigned ctr = 0; ctr < Xsig_pred_.cols(); ++ctr) {
		x_ = x_ + weights_(ctr)*Xsig_pred_.col(ctr);
	}

	//Predict covariance
	P_ = MatrixXd::Zero(P_.rows(), P_.cols());
	MatrixXd diffx = MatrixXd(x_.size(), 1);
	for (ctr = 0; ctr < Xsig_pred_.cols(); ++ctr) {
		diffx = Xsig_pred_.col(ctr) - x_;
		P_ = P_ + weights_(ctr) * diffx * diffx.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
	//Get actual measurement
	VectorXd z = VectorXd(2);
	z << meas_package.raw_measurements_(0),
		 meas_package.raw_measurements_(1);

	//Get z sigma points
	MatrixXd Zsig = MatrixXd(z.size(), Xsig_pred_.cols());
	Zsig << Xsig_pred_.row(0),
			Xsig_pred_.row(1);

	//Predicted measurement mean
	VectorXd z_pred = VectorXd(2);
	z_pred.fill(0.0);
	unsigned ctr;
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		z_pred = z_pred + weights_(ctr) * Zsig.col(ctr);
	}

	//Predicted S matrix
	MatrixXd S = MatrixXd(2, 2);
	MatrixXd diffx = MatrixXd(x_.size(), 1);
	MatrixXd diffz = MatrixXd(z_pred.size(), 1);
	S << std_laspx_*std_laspx_, 0,
		 0, std_laspy_*std_laspy_;
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		diffz = Zsig.col(ctr) - z_pred;
		S = S + weights_(ctr) * diffz * diffz.transpose();
	}

	//Compute cross correlation matrix
	MatrixXd T = MatrixXd(x_.size(), z_pred.size());
	T.fill(0.0);
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		diffz = Zsig.col(ctr) - z_pred;
		diffx = Xsig_pred_.col(ctr) - x_;
		T = T + weights_(ctr) * diffx * diffz.transpose();
	}

	//Compute Kalman Gain
	MatrixXd K = T * S.inverse();

	//State update
	diffz = z - z_pred;
	x_ = x_ + K*diffz;

	//Covariance matrix update
	P_ = P_ - K*S*K.transpose();

	//Calculate NIS_laser_
	NIS_laser_ = (diffz.transpose() * S.inverse() * diffz)(0);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	unsigned ctr;

	//Get actual measurement
	VectorXd z = VectorXd(3);
	z << meas_package.raw_measurements_(0),
		 meas_package.raw_measurements_(1),
		 meas_package.raw_measurements_(2);

	//Get z sigma points
	MatrixXd Zsig = MatrixXd(z.size(), Xsig_pred_.cols());
	VectorXd px, py, v, psi, psidot, rho, phi, rhodot;
	px = Xsig_pred_.row(0);
	py = Xsig_pred_.row(1);
	v = Xsig_pred_.row(2);
	psi = Xsig_pred_.row(3);
	psidot = Xsig_pred_.row(4);
	rho = VectorXd(px.size());
	phi = VectorXd(px.size());
	rhodot = VectorXd(px.size());
	for (ctr = 0; ctr < phi.size(); ++ctr) {
		rho(ctr) = sqrt(px(ctr)*px(ctr) + py(ctr)*py(ctr));

		phi(ctr) = atan2(py(ctr), px(ctr));
		//phi(ctr) = Tools::NormalizeAngle(phi(ctr));

		rhodot(ctr) = (px(ctr) * cos(psi(ctr)) * v(ctr) + py(ctr) * sin(psi(ctr)) * v(ctr))/rho(ctr);
	}

	Zsig.row(0) = rho;
	Zsig.row(1) = phi;
	Zsig.row(2) = rhodot;

	//Predicted measurement mean
	VectorXd z_pred = VectorXd(3);
	z_pred.fill(0.0);
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		z_pred = z_pred + weights_(ctr) * Zsig.col(ctr);
	}

	//Predicted S matrix
	MatrixXd S = MatrixXd(3, 3);
	S << std_radr_*std_radr_, 0, 0,
		 0, std_radphi_*std_radphi_, 0,
		 0, 0, std_radrd_*std_radrd_;
	MatrixXd diffx = MatrixXd(x_.size(), 1);
	MatrixXd diffz = MatrixXd(z_pred.size(), 1);
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		diffz = Zsig.col(ctr) - z_pred;
		S = S + weights_(ctr) * diffz * diffz.transpose();
	}

	//Compute cross correlation matrix
	MatrixXd T = MatrixXd(x_.size(), z_pred.size());
	T.fill(0.0);
	for (ctr = 0; ctr < Zsig.cols(); ++ctr) {
		diffx = Xsig_pred_.col(ctr) - x_;
		diffz = Zsig.col(ctr) - z_pred;
		T = T + weights_(ctr) * diffx * diffz.transpose();
	}

	//Compute Kalman Gain
	MatrixXd K = T * S.inverse();

	//State update
	diffz = z - z_pred;
	x_ = x_ + K*diffz;

	//Covariance matrix update
	P_ = P_ - K*S*K.transpose();

	//Calculate NIS_radar_
	NIS_radar_ = (diffz.transpose() * S.inverse() * diffz)(0);
}
