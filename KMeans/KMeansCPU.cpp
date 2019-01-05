#include "KMeansCPU.h"

KMeansCPU::~KMeansCPU()
{
	delete[] points;
	delete[] result;
}

KMeansCPU::KMeansCPU(Point* points, int size, int k): size(size), k(k)
{
	this->points = new Point[size];
	this->clusters = new int[size];

	for (int i = 0; i < size; i++)
	{
		this->points[i] = points[i];
	}

	this->result = new Point[k];
	this->clusterCounts = new int[k];
}

Point* KMeansCPU::GetResult()
{
	Point* resultCopy = new Point[k];
	for (int i = 0; i < k; i++)
	{
		resultCopy[i] = result[i];
	}

	return resultCopy;
}

void KMeansCPU::Solve()
{
	if (k > size)
		return;

	for (int i = 0; i < size; i++)
	{
		clusters[i] = -1;
	}

	for (int i = 0; i < k; i++)
	{
		result[i] = points[i];
		clusters[i] = i;
	}

	int iteration = 0;
	int delta;
	do
	{
		for (int i = 0; i < k; i++)
		{
			clusterCounts[i] = 0;
		}

		delta = 0;
		for (int i = 0; i < size; i++)
		{
			double minDist = DBL_MAX;
			int bestCluster = -1;

			for (int j = 0; j < k; j++)
			{
				int dist = Distance(points[i], result[j]);
				if (dist < minDist)
				{
					minDist = dist;
					bestCluster = j;
				}
			}

			if (bestCluster != clusters[i])
			{
				clusters[i] = bestCluster;
				delta++;
			}
			clusterCounts[bestCluster]++;
		}

		Point* newResult = new Point[k];

		for (int i = 0; i < size; i++)
		{
			int cluster = clusters[i];
			newResult[cluster] = newResult[cluster] + points[i];
		}

		for (int i = 0; i < k; i++)
		{
			if (clusterCounts[i] == 0)
			{
				continue;
			}
			newResult[i] = newResult[i] / clusterCounts[i];
		}
		delete[] result;
		result = newResult;

		printf("Iteration: %d, delta: %d\n", iteration++, delta);
	} while (delta > 0);
}
