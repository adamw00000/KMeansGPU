#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.h"
#include "KMeansCPU.h"

#include <stdio.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

struct SumPoints: public thrust::binary_function<Point, Point, Point>
{
	__host__ __device__ Point operator()(Point p1, Point p2) { return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z); }
};

__device__ double PointDistance(const Point& p1, const Point& p2)
{
	return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
}

__global__ void FindCluster(Point* d_points, int* d_clusters, Point* d_result, int* d_delta, int* d_k, int* d_size)
{
	int k = *d_k;
	int size = *d_size;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= size)
		return;

	double minDist = DBL_MAX;
	int bestCluster = -1; 
	d_delta[i] = 0;

	for (int j = 0; j < k; j++)
	{
		int dist = PointDistance(d_points[i], d_result[j]);
		if (dist < minDist)
		{
			minDist = dist;
			bestCluster = j;
		}
	}

	//printf("Thread: %d, old cluster: %d, new cluster: %d\n", i, d_clusters[i], bestCluster);

	if (bestCluster != d_clusters[i])
	{
		d_clusters[i] = bestCluster;
		d_delta[i] = 1;
	}
}

__global__ void CalculateResult(Point* d_reduce_result_values, int* d_reduce_count_values, Point* d_result, int* d_k)
{
	int k = *d_k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= k)
		return;

	Point p = d_reduce_result_values[i];
	int count = d_reduce_count_values[i];

	d_result[i] = Point(p.X / count, p.Y / count, p.Z / count);
}

cudaError_t SolveGPU(Point* points, int size, int k);

int main()
{
	std::clock_t c_start, c_end;
	double time_elapsed_ms;

	//int size = 3;
	//int k = 2;

	//Point* points = new Point[size];
	//points[0] = Point(10, 10, 10);
	//points[1] = Point(10, 20, 10);
	//points[2] = Point(20, 10, 10);


	srand(time(NULL));
	int size = 1000000;
	int k = 2;

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

	Point* d_points;
	int* h_clusters;
	int* d_clusters;
	int* d_clusters_copy;
	Point* h_result;
	Point* d_result;
	int* d_k;
	int* d_size;
	int* d_delta;
	int* d_reduce_count_keys;
	int* d_reduce_count_values;
	int* d_reduce_result_keys;
	Point* d_reduce_result_values;
	Point* d_new_result;
	Point* d_points_copy;

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

	cudaStatus = cudaMalloc((void**)&d_reduce_count_keys, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_reduce_count_values, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_reduce_result_keys, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_reduce_result_values, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_new_result, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_points_copy, size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	int nBlocks = size / 1024;
	nBlocks += (size % 1024 == 0) ? 0 : 1;
	int kBlocks = k / 1024;
	kBlocks += (k % 1024 == 0) ? 0 : 1;
	int iteration = 0;

	while (true)
	{
		FindCluster<<<nBlocks, 1024>>>(d_points, d_clusters, d_result, d_delta, d_k, d_size);
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

		thrust::device_ptr<int> dev_clusters_ptr(d_clusters);
		thrust::device_ptr<int> dev_clusters_copy_ptr(d_clusters_copy);
		thrust::device_ptr<int> dev_reduce_count_keys_ptr(d_reduce_count_keys);
		thrust::device_ptr<int> dev_reduce_count_values_ptr(d_reduce_count_values);
		thrust::device_ptr<Point> dev_points_ptr(d_points);
		thrust::device_ptr<Point> dev_points_copy_ptr(d_points_copy);
		thrust::device_ptr<int> dev_reduce_result_keys_ptr(d_reduce_result_keys);
		thrust::device_ptr<Point> dev_reduce_result_values_ptr(d_reduce_result_values);
		thrust::device_ptr<int> dev_delta_ptr(d_delta);

		thrust::copy(thrust::device, dev_clusters_ptr, dev_clusters_ptr + size, dev_clusters_copy_ptr);
		thrust::copy(thrust::device, dev_points_ptr, dev_points_ptr + size, dev_points_copy_ptr);

		thrust::sort_by_key(thrust::device, dev_clusters_copy_ptr, dev_clusters_copy_ptr + size, dev_points_copy_ptr);
		thrust::reduce_by_key(thrust::device, dev_clusters_copy_ptr, dev_clusters_copy_ptr + size, thrust::make_constant_iterator(1), 
			dev_reduce_count_keys_ptr, dev_reduce_count_values_ptr);
		thrust::sort_by_key(thrust::device, dev_reduce_count_keys_ptr, dev_reduce_count_keys_ptr + k, dev_reduce_count_values_ptr);

		thrust::reduce_by_key(thrust::device, dev_clusters_copy_ptr, dev_clusters_copy_ptr + size, dev_points_copy_ptr, 
			dev_reduce_result_keys_ptr, dev_reduce_result_values_ptr, thrust::equal_to<int>(), SumPoints());
		thrust::sort_by_key(thrust::device, dev_reduce_result_keys_ptr, dev_reduce_result_keys_ptr + k, dev_reduce_result_values_ptr);

		CalculateResult<<<kBlocks,1024>>>(d_reduce_result_values, d_reduce_count_values, d_result, d_k);
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

		int delta = thrust::reduce(thrust::device, dev_delta_ptr, dev_delta_ptr + size);
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