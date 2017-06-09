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

	diff_err = 0.0;
	int_err = 0.0;
	total_err = 0.0;
	prev_err = 0.0;

	count = 0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
}

void PID::UpdateError(double err) {
	this->int_err += err;
	if (this->count == 0) {
		this->diff_err = 0.0;
		this->prev_err = err;
	}
	else {
		this->diff_err = err - this->prev_err;
		this->prev_err = err;
	}
	this->count = this->count + 1;
}

double PID::TotalError() {
	return this->int_err;
}

