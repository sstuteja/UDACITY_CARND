#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

  Eigen::MatrixXd GenerateSigmaPointMatrix(const Eigen::VectorXd &x, const Eigen::MatrixXd &P, double lambda);

  static double NormalizeAngle(double angle_radians);

};

#endif /* TOOLS_H_ */
