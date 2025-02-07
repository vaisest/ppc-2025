#include <cmath>
#include <memory>
#include <vector>
#include <iostream>
#include <immintrin.h>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
constexpr int LANES = 16;
typedef float f32x16 __attribute__((vector_size(LANES * sizeof(float))));
constexpr f32x16 zero_vec = {};

double vec_sum(f32x16 v)
{
#ifdef __AVX512DQ__
    return _mm512_reduce_add_ps(v);
#else
    return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])) + ((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15]));
#endif
}

inline f32x16 vec_fmadd(f32x16 a, f32x16 b, f32x16 c)
{
#ifdef __AVX512DQ__
    return _mm512_fmadd_ps(a, b, c);
#else
    // this might get compiled to the above anyway on avx512, but idk let's be
    // sure
    return (a * b) + c;
#endif
}

void correlate(const int ny, const int nx, const float *data, float *result)
{
    // if nx is not divisible by LANES, there's going to be an additional overflow vector
    int row_blocks = nx / LANES + (nx % LANES != 0);
    // width padding multiplier
    constexpr int N = 12;
    // height padding multiplier
    constexpr int M = 14;

    row_blocks = row_blocks + (N - (row_blocks % N));
    const int rows = ny + (M - (ny % M));
    std::unique_ptr<f32x16[]> norm(new f32x16[row_blocks * rows]);
    // approach here is as suggested in the tip

#pragma omp parallel for
    for (int y = 0; y < rows; y++)
    {
        float sum = 0;
        f32x16 arr = zero_vec;

        int counter = 0;
        if (y >= ny)
        {
            for (int i = 0; i < row_blocks; i++)
            {
                norm[y * row_blocks + i] = zero_vec;
            }
            continue;
        }
        for (int block = 0; block < row_blocks; block++)
        {
            for (int v = 0; v < LANES; v++)
            {
                int source_x = block * LANES + v;
                const float value = source_x >= nx ? 0.0 : data[source_x + y * nx];
                arr[counter] = value;
                sum += value;
                counter++;
                if (counter == LANES)
                {
                    norm[y * row_blocks + block] = arr;
                    for (int k = 0; k < LANES; k++)
                    {
                        arr[k] = 0;
                    }

                    counter = 0;
                }
            }
        }

        if (counter != 0)
        {
            norm[y * row_blocks + row_blocks - 1] = arr;
        }

        float mean = sum / (float)nx;
        float sum_sq = 0;
        for (int block = 0; block < row_blocks; block++)
        {
            const int x = block * LANES;
            const int block_idx = y * row_blocks + block;

            // last potentially partially filled block to avoid changing padded
            // zeroes
            if (nx - x < LANES)
            {
                for (int i = 0; i < nx - x; i++)
                {
                    norm[block_idx][i] -= mean;
                }
            }
            else [[likely]]
            {
                norm[block_idx] -= mean;
            }
            f32x16 square = norm[block_idx] * norm[block_idx];
            sum_sq += vec_sum(square);
        }

        float sq_sqrt = sqrt(sum_sq);
        for (int x = 0; x < nx; x += LANES)
        {
            int block_idx = y * row_blocks + x / LANES;
            norm[block_idx] /= sq_sqrt;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < ny; x += M)
    {
        for (int y = 0; y <= x; y += M)
        {
            // we calculate MxM row correlations at a time
            f32x16 stuff[M][M] = {};

            // avoid calculating unnecessary correlations
            const int iM = M > ny - y ? ny - y : M;
            const int jM = M > ny - x ? ny - x : M;

            for (int k = 0; k < row_blocks; k += N)
            {
                for (int i = 0; i < iM; i++)
                {
                    f32x16 y0 = norm[(y + i) * row_blocks + k];
                    f32x16 y1 = norm[(y + i) * row_blocks + k + 1];
                    f32x16 y2 = norm[(y + i) * row_blocks + k + 2];
                    f32x16 y3 = norm[(y + i) * row_blocks + k + 3];
                    f32x16 y4 = norm[(y + i) * row_blocks + k + 4];
                    f32x16 y5 = norm[(y + i) * row_blocks + k + 5];
                    f32x16 y6 = norm[(y + i) * row_blocks + k + 6];
                    f32x16 y7 = norm[(y + i) * row_blocks + k + 7];
                    f32x16 y8 = norm[(y + i) * row_blocks + k + 8];
                    f32x16 y9 = norm[(y + i) * row_blocks + k + 9];
                    f32x16 y10 = norm[(y + i) * row_blocks + k + 10];
                    f32x16 y11 = norm[(y + i) * row_blocks + k + 11];
                    // f64x8 y12 = norm[(y + i) * row_blocks + k + 12];
                    // f64x8 y13 = norm[(y + i) * row_blocks + k + 13];
                    // f64x8 y14 = norm[(y + i) * row_blocks + k + 14];
                    // f64x8 y15 = norm[(y + i) * row_blocks + k + 15];
                    // f64x8 y16 = norm[(y + i) * row_blocks + k + 16];
                    // f64x8 y17 = norm[(y + i) * row_blocks + k + 17];
                    for (int j = 0; j < jM; j++)
                    {
                        f32x16 x0 = norm[(x + j) * row_blocks + k];
                        f32x16 x1 = norm[(x + j) * row_blocks + k + 1];
                        f32x16 x2 = norm[(x + j) * row_blocks + k + 2];
                        f32x16 x3 = norm[(x + j) * row_blocks + k + 3];
                        f32x16 x4 = norm[(x + j) * row_blocks + k + 4];
                        f32x16 x5 = norm[(x + j) * row_blocks + k + 5];
                        f32x16 x6 = norm[(x + j) * row_blocks + k + 6];
                        f32x16 x7 = norm[(x + j) * row_blocks + k + 7];
                        f32x16 x8 = norm[(x + j) * row_blocks + k + 8];
                        f32x16 x9 = norm[(x + j) * row_blocks + k + 9];
                        f32x16 x10 = norm[(x + j) * row_blocks + k + 10];
                        f32x16 x11 = norm[(x + j) * row_blocks + k + 11];
                        // f64x8 x12 = norm[(x + j) * row_blocks + k + 12];
                        // f64x8 x13 = norm[(x + j) * row_blocks + k + 13];
                        // f64x8 x14 = norm[(x + j) * row_blocks + k + 14];
                        // f64x8 x15 = norm[(x + j) * row_blocks + k + 15];
                        // f64x8 x16 = norm[(x + j) * row_blocks + k + 16];
                        // f64x8 x17 = norm[(x + j) * row_blocks + k + 17];

                        x0 *= y0;
                        x2 *= y2;

                        x0 = vec_fmadd(y1, x1, x0);
                        x2 = vec_fmadd(y3, x3, x2);
                        x4 *= y4;
                        x6 *= y6;
                        x4 = vec_fmadd(y5, x5, x4);
                        x6 = vec_fmadd(y7, x7, x6);

                        x8 *= y8;
                        x10 *= y10;
                        // x12 *= y12;
                        // x14 *= y14;
                        // x16 *= y16;

                        x8 = vec_fmadd(y9, x9, x8);
                        x10 = vec_fmadd(y11, x11, x10);
                        // x12 = vec_fmadd(y13, x13, x12);
                        // x14 = vec_fmadd(y15, x15, x14);
                        // x16 = vec_fmadd(y17, x17, x16);

                        stuff[i][j] += ((x0 + x2) + (x4 + x6)) + ((x8 + x10));
                    }
                }
            }

            for (int i = 0; i < iM; i++)
            {
                for (int j = 0; j < jM; j++)
                {
                    result[(y + i) * ny + x + j] = vec_sum(stuff[i][j]);
                }
            }
        }
    }
}