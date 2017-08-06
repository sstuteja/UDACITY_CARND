#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// TODO - complete this function
vector<double> JMT(vector<double> start, vector <double> end, double T)
{
    /*
    Calculate the Jerk Minimizing Trajectory that connects the initial state
    to the final state in time T.

    INPUTS

    start - the vehicles start location given as a length three array
        corresponding to initial values of [s, s_dot, s_double_dot]

    end   - the desired end state for vehicle. Like "start" this is a
        length three array.

    T     - The duration, in seconds, over which this maneuver should occur.

    OUTPUT 
    an array of length 6, each value corresponding to a coefficent in the polynomial 
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

    EXAMPLE

    > JMT( [0, 10, 0], [10, 10, 0], 1)
    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    */
    MatrixXd TMatrix(3, 3);
    TMatrix << pow(T, 3), pow(T, 4), pow(T, 5),
              3.0*pow(T, 2), 4.0*pow(T, 3), 5.0*pow(T, 4),
              6.0*T, 12.0*pow(T, 2), 20.0*pow(T, 3);
              
    double si = start[0];
    double sidot = start[1];
    double sidotdot = start[2];
    
    double sf = end[0];
    double sfdot = end[1];
    double sfdotdot = end[2];
    
    MatrixXd OutMatrix(3, 1);
    OutMatrix << sf - (si + sidot*T + 0.5*sidotdot*T*T),
                 sfdot - (sidot + sidotdot*T),
                 sfdotdot - sidotdot;
    
    MatrixXd Solution(3, 1);
    Solution = TMatrix.inverse() * OutMatrix;
    
    double alpha0 = si;
    double alpha1 = sidot;
    double alpha2 = 0.5*sidotdot;
    double alpha3 = Solution(0, 0);
    double alpha4 = Solution(1, 0);
    double alpha5 = Solution(2, 0);
    
    return {alpha0, alpha1, alpha2, alpha3, alpha4, alpha5};
    
}

bool close_enough(vector< double > poly, vector<double> target_poly, double eps=0.01) {


	if(poly.size() != target_poly.size())
	{
		cout << "your solution didn't have the correct number of terms" << endl;
		return false;
	}
	for(int i = 0; i < poly.size(); i++)
	{
		double diff = poly[i]-target_poly[i];
		if(abs(diff) > eps)
		{
			cout << "at least one of your terms differed from target by more than " << eps << endl;
			return false;
		}

	}
	return true;
}
	
struct test_case {
	
		vector<double> start;
		vector<double> end;
		double T;
};

vector< vector<double> > answers = {{0.0, 10.0, 0.0, 0.0, 0.0, 0.0},{0.0,10.0,0.0,0.0,-0.625,0.3125},{5.0,10.0,1.0,-3.0,0.64,-0.0432}};

int main() {

	//create test cases

	vector< test_case > tc;

	test_case tc1;
	tc1.start = {0,10,0};
	tc1.end = {10,10,0};
	tc1.T = 1;
	tc.push_back(tc1);

	test_case tc2;
	tc2.start = {0,10,0};
	tc2.end = {20,15,20};
	tc2.T = 2;
	tc.push_back(tc2);

	test_case tc3;
	tc3.start = {5,10,2};
	tc3.end = {-30,-20,-4};
	tc3.T = 5;
	tc.push_back(tc3);

	bool total_correct = true;
	for(int i = 0; i < tc.size(); i++)
	{
		vector< double > jmt = JMT(tc[i].start, tc[i].end, tc[i].T);
		bool correct = close_enough(jmt,answers[i]);
		total_correct &= correct;

	}
	if(!total_correct)
	{
		cout << "Try again!" << endl;
	}
	else
	{
		cout << "Nice work!" << endl;
	}

	return 0;
}