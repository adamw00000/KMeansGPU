#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <float.h>

#define POINTS_PER_THREAD 32
//#define TIMERS 1

class Point
{
public:
	double X, Y, Z;

	__host__ __device__ Point() : X(0), Y(0), Z(0) {}
	__host__ __device__ Point(double x, double y, double z) : X(x), Y(y), Z(z) {}

	friend std::ostream & operator<<(std::ostream & ostream, const Point& p)
	{
		ostream << "(" << p.X << ", " << p.Y << ", " << p.Z << ")";
		return ostream;
	}

	friend Point operator+(const Point & p1, const Point & p2)
	{
		return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z);
	}

	friend Point operator-(const Point & p1, const Point & p2)
	{
		return Point(p1.X - p2.X, p1.Y - p2.Y, p1.Z - p2.Z);
	}

	friend Point operator*(const Point & p1, const double & a)
	{
		return Point(p1.X * a, p1.Y * a, p1.Z * a);
	}

	friend Point operator*(const double & a, const Point & p1)
	{
		return Point(p1.X * a, p1.Y * a, p1.Z * a);
	}

	friend Point operator/(const Point & p1, const double & a)
	{
		return Point(p1.X / a, p1.Y / a, p1.Z / a);
	}

	friend double Distance(const Point & p1, const Point & p2)
	{
		return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
	}
};

class KMeansCPU
{
	Point* points;
	int* clusters;
	int* clusterCounts;
	Point* result;
	int size;
	int k;
public:
	KMeansCPU(Point* points, int size, int k) : size(size), k(k)
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
	~KMeansCPU()
	{
		delete[] points;
		delete[] result;
	}

	Point* GetResult()
	{
		Point* resultCopy = new Point[k];
		for (int i = 0; i < k; i++)
		{
			resultCopy[i] = result[i];
		}

		return resultCopy;
	}
	void Solve()
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
};

struct SumPoints: public thrust::binary_function<Point, Point, Point>
{
	__host__ __device__ Point operator()(Point p1, Point p2) { return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z); }
};

__device__ double PointDistance(const Point& p1, const Point& p2)
{
	return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
}

__global__ void Reset(Point* d_new_result, int* d_counts, int* d_k)
{
	int k = *d_k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= k)
		return;

	d_counts[i] = 0;
	d_new_result[i] = Point();
}

__global__ void FindCluster(Point* d_points, int* d_clusters, Point* d_result, Point* d_new_result, int* d_counts, int* d_delta, int* d_k, int* d_size, int threads)
{
	int k = *d_k;
	int size = *d_size;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int cluster = 0; cluster < k; cluster++)
	{
		d_new_result[cluster * threads + i] = Point();
		d_counts[cluster * threads + i] = 0;
	}

	for (int pointNr = 0; pointNr < POINTS_PER_THREAD; pointNr++)
	{
		int index = i * POINTS_PER_THREAD + pointNr;

		if (index >= size)
			return;

		double minDist = DBL_MAX;
		int bestCluster = -1;
		d_delta[index] = 0;

		Point p = d_points[index];

		for (int j = 0; j < k; j++)
		{
			int dist = PointDistance(p, d_result[j]);
			if (dist < minDist)
			{
				minDist = dist;
				bestCluster = j;
			}
		}

		//printf("Index: %d, old cluster: %d, new cluster: %d\n", index, d_clusters[index], bestCluster);

		if (bestCluster != d_clusters[index])
		{
			d_clusters[index] = bestCluster;
			d_delta[index] = 1;
		}

		Point oldNewResult = d_new_result[bestCluster * threads + i];
		d_new_result[bestCluster * threads + i] = Point(oldNewResult.X + p.X, oldNewResult.Y + p.Y, oldNewResult.Z + p.Z);
		d_counts[bestCluster * threads + i]++;
	}
}

__global__ void CalculateResult(Point* d_new_result, int* d_counts, Point* d_result, int* d_k)
{
	int k = *d_k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= k)
		return;

	Point p = d_new_result[i];
	int count = d_counts[i];

	d_result[i] = Point(p.X / count, p.Y / count, p.Z / count);
}

cudaError_t SolveGPU(Point* points, int size, int k);

int main()
{
	std::clock_t c_start, c_end;
	double time_elapsed_ms;
/*
	int size = 3;
	int k = 2;

	Point* points = new Point[size];
	points[0] = Point(10, 10, 10);
	points[1] = Point(10, 20, 10);
	points[2] = Point(20, 10, 10);*/


	srand(time(NULL));
	int size = 1000000;
	int k = 10;

	Point* points = new Point[size];
	for (int i = 0; i < size; i++)
	{
		points[i] = Point(rand() % 1000, rand() % 1000, rand() % 1000);
	}

	c_start = std::clock();
	if (SolveGPU(points, size, k) != cudaSuccess) 
	{
		fprintf(stderr, "SolveGPU failed!");
		
		delete[] points;
		return 1;
	}
	c_end = std::clock();
	time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	printf("Time for GPU: %lf ms\n", time_elapsed_ms);
	std::cout << std::endl;

	c_start = std::clock();
	KMeansCPU solver = KMeansCPU(points, size, k);
	solver.Solve();

	std::cout << "CPU result:" << std::endl;
	auto result = solver.GetResult();
	for (int i = 0; i < k; i++)
		std::cout << result[i] << std::endl;

	c_end = std::clock();
	time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	printf("Time for CPU: %lf ms\n", time_elapsed_ms);

	delete[] points;
	delete[] result;
    return 0;
}

cudaError_t SolveGPU(Point* h_points, int size, int k)
{
	cudaError_t cudaStatus = cudaSuccess;
#ifdef TIMERS
	std::clock_t c_start, c_end;
	double time_elapsed_ms;
#endif

	Point* d_points;
	int* h_clusters;
	int* d_clusters;
	int* d_clusters_copy;
	Point* h_result;
	Point* d_result;
	Point* d_new_result;
	Point* d_new_result_final;
	int* d_counts;
	int* d_counts_final;
	int* d_k;
	int* d_size;
	int* d_delta;

	int nThreads = 128;
	int nBlocks = size / POINTS_PER_THREAD / nThreads;
	nBlocks += (size % nThreads == 0) ? 0 : 1;
	int kBlocks = k / nThreads;
	kBlocks += (k % nThreads == 0) ? 0 : 1;
	int iteration = 0;

	h_clusters = new int[size];
	h_result = new Point[k];

	for (int i = 0; i < size; i++)
	{
		h_clusters[i] = -1;
	}

	for (int i = 0; i < k; i++)
	{
		h_result[i] = h_points[i];
		h_clusters[i] = i;
	}

	cudaStatus = cudaMalloc((void**)&d_points, size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMalloc failed!");
	    goto ErrorStart;
	}

	cudaStatus = cudaMemcpy(d_points, h_points, size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error0;
	}

	cudaStatus = cudaMalloc((void**)&d_clusters, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error0;
	}

	cudaStatus = cudaMemcpy(d_clusters, h_clusters, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&d_clusters_copy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&d_result, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error2;
	}

	cudaStatus = cudaMemcpy(d_result, h_result, k * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error3;
	}

	cudaStatus = cudaMalloc((void**)&d_k, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error3;
	}

	cudaStatus = cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error4;
	}

	cudaStatus = cudaMalloc((void**)&d_size, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error4;
	}

	cudaStatus = cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error5;
	}

	cudaStatus = cudaMalloc((void**)&d_delta, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error5;
	}

	cudaStatus = cudaMalloc((void**)&d_new_result, nThreads * nBlocks * k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error6;
	}

	cudaStatus = cudaMalloc((void**)&d_new_result_final, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error7;
	}

	cudaStatus = cudaMalloc((void**)&d_counts, nThreads * nBlocks * k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error8;
	}

	cudaStatus = cudaMalloc((void**)&d_counts_final, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error9;
	}

	while (true)
	{
#ifdef TIMERS
		c_start = std::clock();
#endif
		Reset<<<kBlocks, nThreads>>>(d_new_result_final, d_counts_final, d_k);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "ResetCounts launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ResetCounts!\n", cudaStatus);
				goto Error;
			}

#ifdef TIMERS
			c_end = std::clock();
			time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
			printf("Time for Reset: %lf ms\n", time_elapsed_ms);
			c_start = std::clock();
#endif
		FindCluster<<<nBlocks, nThreads>>>(d_points, d_clusters, d_result, d_new_result, d_counts, d_delta, d_k, d_size, nBlocks * nThreads);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "FindCluster launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FindCluster!\n", cudaStatus);
				goto Error;
			}
#ifdef TIMERS
			c_end = std::clock();
			time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
			printf("Time for FindCluster: %lf ms\n", time_elapsed_ms);
			c_start = std::clock();
#endif

		for (int i = 0; i < k; i++)
		{
			thrust::device_ptr<Point> dev_new_result_ptr(d_new_result + i * nBlocks * nThreads);
			thrust::device_ptr<int> dev_count_ptr(d_counts + i * nBlocks * nThreads);
			thrust::device_ptr<Point> dev_new_result_final_ptr(d_new_result_final);
			thrust::device_ptr<int> dev_count_final_ptr(d_counts_final);

			dev_new_result_final_ptr[i] = thrust::reduce(dev_new_result_ptr, dev_new_result_ptr + nThreads * nBlocks, Point(), SumPoints());
			dev_count_final_ptr[i] = thrust::reduce(dev_count_ptr, dev_count_ptr + nThreads * nBlocks);
		}
#ifdef TIMERS
		c_end = std::clock();
		time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
		printf("Time for reduces: %lf ms\n", time_elapsed_ms);
		c_start = std::clock();
#endif

		CalculateResult<<<kBlocks, nThreads>>>(d_new_result_final, d_counts_final, d_result, d_k);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "CalculateResult launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CalculateResult!\n", cudaStatus);
				goto Error;
			}
#ifdef TIMERS
			c_end = std::clock();
			time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
			printf("Time for CalculateResult: %lf ms\n", time_elapsed_ms);
			c_start = std::clock();
#endif

		thrust::device_ptr<int> dev_delta_ptr(d_delta);
		int delta = thrust::reduce(thrust::device, dev_delta_ptr, dev_delta_ptr + size);

#ifdef TIMERS
		c_end = std::clock();
		time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
		printf("Time for delta: %lf ms\n", time_elapsed_ms);
#endif
		std::cout << "Iteration: "<< iteration++ << ", delta: " << delta << std::endl;
		if (delta == 0)
			break;
	}

	cudaStatus = cudaMemcpy(h_result, d_result, k * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	std::cout << "GPU result:" << std::endl;
	for (int i = 0; i < k; i++)
	{
		std::cout << h_result[i] << std::endl;
	}
Error:
	cudaFree(d_counts_final);
Error9:
	cudaFree(d_counts);
Error8:
	cudaFree(d_new_result_final);
Error7:
	cudaFree(d_new_result);
Error6:
	cudaFree(d_delta);
Error5:
	cudaFree(d_size);
Error4:
	cudaFree(d_k);
Error3:
	cudaFree(d_result);
Error2:
	cudaFree(d_clusters_copy);
Error1:
	cudaFree(d_clusters);
Error0:
	cudaFree(d_points);

ErrorStart:
	delete[] h_clusters;
	delete[] h_result;

	return cudaStatus;
}