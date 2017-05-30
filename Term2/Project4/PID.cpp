#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {
	p_error = 0.0;
	i_error = 0.0;
	d_error = 0.0;

	Kp = 0.0;
	Ki = 0.0;
	Kd = 0.0;

	diff_cte = 0.0;
	int_cte = 0.0;
	total_err = 0.0;
	prev_cte = 0.0;

	count = 0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
}

void PID::UpdateError(double cte) {
	this->int_cte += cte;
	if (this->count == 0) {
		this->diff_cte = 0.0;
		this->prev_cte = cte;
	}
	else {
		this->diff_cte = cte - this->prev_cte;
		this->prev_cte = cte;
	}
	this->count++;
}

double PID::TotalError() {
	return 0.0;
}

