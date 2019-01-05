#include "Point.h"

__host__ __device__ std::ostream & operator<<(std::ostream & ostream, const Point& p)
{
	ostream << "(" << p.X << ", " << p.Y << ", " << p.Z << ")";
	return ostream;
}

__host__ __device__ Point operator+(const Point & p1, const Point & p2)
{
	return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z);
}

__host__ __device__ Point operator-(const Point & p1, const Point & p2)
{
	return Point(p1.X - p2.X, p1.Y - p2.Y, p1.Z - p2.Z);
}

__host__ __device__ Point operator*(const Point & p1, const double & a)
{
	return Point(p1.X * a, p1.Y * a, p1.Z * a);
}

__host__ __device__ Point operator*(const double & a, const Point & p1)
{
	return operator*(a, p1);
}

__host__ __device__ Point operator/(const Point & p1, const double & a)
{
	return Point(p1.X / a, p1.Y / a, p1.Z / a);
}

__host__ __device__ double Distance(const Point & p1, const Point & p2)
{
	return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
}
