#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>


class Point
{
public:
	double X, Y, Z;

	__host__ __device__ Point(): X(0), Y(0), Z(0) {}
	__host__ __device__ Point(double x, double y, double z) : X(x), Y(y), Z(z) {}

	friend __host__ __device__  Point operator+(const Point& p1, const Point& p2);
	friend __host__ __device__  Point operator-(const Point& p1, const Point& p2);
	friend __host__ __device__  Point operator*(const Point& p1, const double& a);
	friend __host__ __device__  Point operator*(const double& a, const Point& p1);
	friend __host__ __device__  Point operator/(const Point& p1, const double& a);
	friend __host__  double Distance(const Point& p1, const Point& p2);

	friend std::ostream& operator<<(std::ostream& ostream, const Point& p);
};

