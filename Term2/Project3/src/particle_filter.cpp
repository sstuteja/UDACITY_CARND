/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//Initializing number of particles to 250
	num_particles = 250;
	Particle thisParticle;

	std::default_random_engine generator;

	std::normal_distribution<double> dist_x(0.0, std[0]);
	std::normal_distribution<double> dist_y(0.0, std[1]);
	std::normal_distribution<double> dist_theta(0.0, std[2]);

	for (unsigned ctr = 0; ctr < (unsigned)num_particles; ++ctr) {
		thisParticle.x = x + dist_x(generator);
		thisParticle.y = y + dist_y(generator);
		thisParticle.theta = theta + dist_theta(generator);
		thisParticle.weight = 1.0/num_particles;
		thisParticle.id = ctr;

		particles.push_back(thisParticle);
	}

	weights.resize(num_particles, 1.0/num_particles);

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine generator;

	std::normal_distribution<double> dist_x(0.0, std_pos[0]);
	std::normal_distribution<double> dist_y(0.0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (unsigned ctr = 0; ctr < particles.size(); ++ctr) {
		if (fabs(yaw_rate) >= 0.001) {
			particles[ctr].x += (velocity/yaw_rate)*(sin(particles[ctr].theta + yaw_rate*delta_t) - sin(particles[ctr].theta)) + dist_x(generator);
			particles[ctr].y += (velocity/yaw_rate)*(cos(particles[ctr].theta) - cos(particles[ctr].theta + yaw_rate*delta_t)) + dist_y(generator);
			particles[ctr].theta += yaw_rate*delta_t + dist_theta(generator);
		}
		else {
			particles[ctr].x += velocity*delta_t*cos(particles[ctr].theta) + dist_x(generator);
			particles[ctr].y += velocity*delta_t*cos(particles[ctr].theta) + dist_y(generator);
			particles[ctr].theta += dist_theta(generator);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double minDistance;
	double thisDistance;
	int minID;

	for (unsigned obs_ctr = 0; obs_ctr < observations.size(); ++obs_ctr) {
		minDistance = dist(observations[obs_ctr].x, observations[obs_ctr].y, predicted[0].x, predicted[0].y);
		minID = predicted[0].id;
		for (unsigned pred_ctr = 0; pred_ctr < predicted.size(); ++pred_ctr) {
			thisDistance = dist(observations[obs_ctr].x, observations[obs_ctr].y, predicted[pred_ctr].x, predicted[pred_ctr].y);
			if (thisDistance < minDistance) {
				minDistance = thisDistance;
				minID = predicted[pred_ctr].id;
			}
		}
		observations[obs_ctr].id = minID;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double sigx;
	double sigy;
	double devX_sq;
	double devY_sq;
	double WEIGHTS_SUM = 0.0;
	LandmarkObs thisLandmark;

	sigx = std_landmark[0];
	sigy = std_landmark[1];

	for (unsigned ctr = 0; ctr < particles.size(); ++ctr) {
		particles[ctr].weight = 1.0;

		std::vector<LandmarkObs> landmarks_in_range;
		for (unsigned ctr2 = 0; ctr2 < map_landmarks.landmark_list.size(); ++ctr2) {
			if (dist(map_landmarks.landmark_list[ctr2].x_f, map_landmarks.landmark_list[ctr2].y_f, particles[ctr].x, particles[ctr].y) <= sensor_range) {
				thisLandmark.x = map_landmarks.landmark_list[ctr2].x_f;
				thisLandmark.y = map_landmarks.landmark_list[ctr2].y_f;
				thisLandmark.id = map_landmarks.landmark_list[ctr2].id_i;
				landmarks_in_range.push_back(thisLandmark);
			}
		}

		//Transform observations from local coordinates (wrt vehicle) to map coordinates
		//Formula: xp + xo*cos(theta) - yo*sin(theta), yp + xo*sin(theta) + yo*cos(theta)
		std::vector<LandmarkObs> observations_map;
		observations_map.resize(observations.size());
		for (unsigned ctr2 = 0; ctr2 < observations.size(); ++ctr2) {
			observations_map[ctr2].x = particles[ctr].x + observations[ctr2].x * cos(particles[ctr].theta) - observations[ctr2].y * sin(particles[ctr].theta);
			observations_map[ctr2].y = particles[ctr].y + observations[ctr2].x * sin(particles[ctr].theta) + observations[ctr2].y * cos(particles[ctr].theta);
		}

		if (landmarks_in_range.size() >= 1) {
			//Store the ID of nearest landmark inside observations_map
			dataAssociation(landmarks_in_range, observations_map);

			//Use multivariate normal distribution to update the weights for each observation
			LandmarkObs ClosestLandmark;
			for (unsigned ctr2 = 0; ctr2 < observations_map.size(); ++ctr2) {
				//Find corresponding landmark for this particular ID
				for (unsigned ctr3 = 0; ctr3 < landmarks_in_range.size(); ++ctr3) {
					if (landmarks_in_range[ctr3].id == observations_map[ctr2].id) {
						ClosestLandmark = landmarks_in_range[ctr3];
						devX_sq = (observations_map[ctr2].x - ClosestLandmark.x)*(observations_map[ctr2].x - ClosestLandmark.x);
						devY_sq = (observations_map[ctr2].y - ClosestLandmark.y)*(observations_map[ctr2].y - ClosestLandmark.y);
						particles[ctr].weight *= (1.0/(2*M_PI*sigx*sigy)) * exp(-( (devX_sq/(2*sigx*sigx)) + (devY_sq/(2*sigy*sigy)) ));
						break;
					}
				}
			}
			weights[ctr] = particles[ctr].weight;

			WEIGHTS_SUM += weights[ctr];
		}
		else {
			particles[ctr].weight = 0.0;
			weights[ctr] = 0.0;
		}
	}

	//Normalize
	for (unsigned ctr = 0; ctr < weights.size(); ++ctr) {
		weights[ctr] /= WEIGHTS_SUM;
		particles[ctr].weight /= WEIGHTS_SUM;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	std::vector<Particle> particles_resampled;

	particles_resampled.resize(num_particles);

	std::discrete_distribution<> distribution(weights.begin(), weights.end());

	for (unsigned ctr = 0; ctr < (unsigned)num_particles; ++ctr) {
		particles_resampled[ctr] = particles[distribution(gen)];
	}

	particles = particles_resampled;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
