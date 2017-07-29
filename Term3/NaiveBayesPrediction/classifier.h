#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);
private:
    vector<double> mean_s;
    vector<double> mean_sdot;
    vector<double> mean_d;
    vector<double> mean_ddot;

    vector<double> std_s;
    vector<double> std_sdot;
    vector<double> std_d;
    vector<double> std_ddot;
};

double gaussian_prob(double mu, double sig, double o);

#endif



