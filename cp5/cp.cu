#include <cmath>
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <ranges>
#include <numeric>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

__global__ void sum_kernel(const float *data, float *row_sums, const int ny, const int nx, const bool square)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (row >= ny)
    {
        return;
    }
    for (int i = row; i < ny; i += stride)
    {
        float sum = 0.0;
        for (int x = 0; x < nx; x++)
        {
            const float val = data[i * nx + x];
            sum += square ? val * val : val;
        }
        row_sums[i] = square ? sqrt(sum) : sum / nx;
    }
}
__global__ void mean_sub_kernel(float *data, float *row_sums, const int ny, const int nx)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < ny * nx; i += stride)
    {
        int row = i / nx;
        data[i] -= row_sums[row];
    }
}
__global__ void sqrt_div_kernel(float *data, const float *row_sums, const int ny, const int nx)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < ny * nx; i += stride)
    {
        int row = i / nx;
        data[i] /= row_sums[row];
    }
}
__global__ void matmul_kernel(float *data, float *result, const int ny, const int nx)
{
    constexpr size_t M = 8;
    constexpr size_t N = 4;
    int y = (blockIdx.x * blockDim.x + threadIdx.x) * M;
    int x = (blockIdx.y * blockDim.y + threadIdx.y) * M;
    if (y >= ny || x >= ny)
    {
        return;
    }

    float stuff[M][M] = {};
    // const auto rem = nx % 16;
    // limit iteration length to within bounds
    const int iM = M > ny - y ? ny - y : M;
    const int jM = M > ny - x ? ny - x : M;
    if (y <= x) [[likely]]
    {
        const auto rem = nx % N;
        for (size_t k = 0; k < nx - rem; k += N)
        {
            for (size_t i = 0; i < iM; i++)
            {
                float ys[N];
                for (size_t v = 0; v < N; v++)
                {
                    ys[v] = data[(y + i) * nx + k + v];
                }
                for (size_t j = 0; j < jM; j++)
                {
                    for (size_t v = 0; v < N; v++)
                    {
                        stuff[i][j] += ys[v] * data[(x + j) * nx + k + v];
                    }
                }
            }
        }
        for (size_t k = nx - rem; k < nx; k++)
        {
            for (size_t i = 0; i < iM; i++)
            {
                for (size_t j = 0; j < jM; j++)
                {
                    stuff[i][j] += data[(y + i) * nx + k] * data[(x + j) * nx + k];
                }
            }
        }
    }
    for (size_t i = 0; i < iM; i++)
    {
        for (size_t j = 0; j < jM; j++)
        {
            result[(y + i) * ny + x + j] = stuff[i][j];
        }
    }
}
int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

void correlate(int ny, int nx, const float *data, float *result)
{
    // input data
    float *dataGPU = NULL;
    CHECK(cudaMalloc(&dataGPU, ny * nx * sizeof(float)));
    // row sums, and squared sums used for normalising input
    float *rowsGPU = NULL;
    CHECK(cudaMalloc(&rowsGPU, ny * sizeof(float)));
    // output
    float *resGPU = NULL;
    CHECK(cudaMalloc(&resGPU, ny * ny * sizeof(float)));

    // copy input
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int num_blocks = div_up(ny * nx, block_size);
    int num_blocks_ny = div_up(ny, block_size);
    // get average of rows
    sum_kernel<<<num_blocks_ny, block_size>>>(dataGPU, rowsGPU, ny, nx, false);
    CHECK(cudaGetLastError());

    // subtract average of rows from each row
    mean_sub_kernel<<<num_blocks, block_size>>>(dataGPU, rowsGPU, ny, nx);
    CHECK(cudaGetLastError());

    // get sum of squares of rows
    sum_kernel<<<num_blocks_ny, block_size>>>(dataGPU, rowsGPU, ny, nx, true);
    CHECK(cudaGetLastError());

    // divide rows by sum of squares of rows
    sqrt_div_kernel<<<num_blocks, block_size>>>(dataGPU, rowsGPU, ny, nx);
    CHECK(cudaGetLastError());

    // performance at larger block sizes seems worse for some reason

    int size = 16;
    dim3 block(size, size);
    dim3 grid(div_up(ny, size), div_up(ny, size));

    // calculate AA^T
    matmul_kernel<<<grid, block>>>(dataGPU, resGPU, ny, nx);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(resGPU));
    CHECK(cudaFree(rowsGPU));
}
