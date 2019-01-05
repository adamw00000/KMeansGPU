#pragma once
#include "Point.h"
#include <float.h>

class KMeansCPU
{
	Point* points;
	int* clusters;
	int* clusterCounts;
	Point* result;
	int size;
	int k;
public:
	KMeansCPU(Point* points, int size, int k);
	~KMeansCPU();

	Point* GetResult();
	void Solve();
};

