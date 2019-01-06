#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.h"
#include "KMeansCPU.h"

#include <stdio.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define POINTS_PER_THREAD 32
//#define TIMERS 1

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
	cudaError_t cudaStatus;
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
	    goto Error;
	}

	cudaStatus = cudaMemcpy(d_points, h_points, size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_clusters, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_clusters_copy, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_clusters, h_clusters, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_result, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_result, h_result, k * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_k, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_size, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_delta, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_new_result, nThreads * nBlocks * k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_new_result_final, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_counts, nThreads * nBlocks * k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_counts_final, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
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
	delete[] h_clusters;
	delete[] h_result;
	cudaFree(d_points);
	cudaFree(d_clusters);
	cudaFree(d_result);

	return cudaSuccess;
}