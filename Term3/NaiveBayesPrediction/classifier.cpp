#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d,
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

	mean_s.resize(0);
	mean_sdot.resize(0);
	mean_d.resize(0);
	mean_ddot.resize(0);
	std_s.resize(0);
	std_sdot.resize(0);
	std_d.resize(0);
	std_ddot.resize(0);

	unsigned long long count = 0;
	double sum_s = 0.0;
    double sum_sdot = 0.0;
    double sum_d = 0.0;
    double sum_ddot = 0.0;

	for (unsigned ctr = 0; ctr < possible_labels.size(); ++ctr) {
	    sum_s = 0.0;
	    sum_sdot = 0.0;
	    sum_d = 0.0;
	    sum_ddot = 0.0;
	    count = 0;
	    for (unsigned ctr2 = 0; ctr2 < labels.size(); ++ctr2) {
	        if (labels[ctr2].compare(possible_labels[ctr]) == 0) {
	            sum_s += data[ctr2][0];
	            sum_sdot += data[ctr2][1];
	            sum_d += fmod(data[ctr2][2], 4.0);
	            sum_ddot += data[ctr2][3];

	            count++;
	        }
	    }

	    mean_s.push_back(sum_s/count);
	    mean_sdot.push_back(sum_sdot/count);
	    mean_d.push_back(sum_d/count);
	    mean_ddot.push_back(sum_ddot/count);

	    sum_s = 0.0;
	    sum_sdot = 0.0;
	    sum_d = 0.0;
	    sum_ddot = 0.0;
	    count = 0;
	    for (unsigned ctr2 = 0; ctr2 < labels.size(); ++ctr2) {
	        if (labels[ctr2].compare(possible_labels[ctr]) == 0) {
	            sum_s += pow(data[ctr2][0] - mean_s[ctr], 2.0);
	            sum_sdot += pow(data[ctr2][1] - mean_sdot[ctr], 2.0);
	            sum_d += pow(fmod(data[ctr2][2], 4.0) - mean_d[ctr], 2.0);
	            sum_ddot += pow(data[ctr2][3] - mean_ddot[ctr], 2.0);

	            count++;
	        }
	    }
	    std_s.push_back(sqrt(sum_s/count));
	    std_sdot.push_back(sqrt(sum_sdot/count));
	    std_d.push_back(sqrt(sum_d/count));
	    std_ddot.push_back(sqrt(sum_ddot/count));
	}
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	vector<double> probability_list;
	double product;
	unsigned maxidx;

	for (unsigned ctr = 0; ctr < possible_labels.size(); ++ctr) {
	    product = 1.0;
    	product *= gaussian_prob(mean_s[ctr], std_s[ctr], sample[0]);
    	product *= gaussian_prob(mean_sdot[ctr], std_sdot[ctr], sample[1]);
    	product *= gaussian_prob(mean_d[ctr], std_d[ctr], fmod(sample[2], 4.0));
    	product *= gaussian_prob(mean_ddot[ctr], std_ddot[ctr], sample[3]);
	    probability_list.push_back(product);
	}

	maxidx = 0;
	for (unsigned ctr = 1; ctr < possible_labels.size(); ++ctr) {
	    if (probability_list[ctr] > probability_list[maxidx]) {
	        maxidx = ctr;
	    }
	}

	return possible_labels[maxidx];
}

double gaussian_prob(double mu, double sig, double obs) {
    double num, denum, norm;
    num = pow(obs - mu, 2.0);
    denum = 2.0*sig*sig;
    norm = 1.0 / sqrt(2.0*M_PI*sig*sig);
    return norm * exp(-num/denum);
}
