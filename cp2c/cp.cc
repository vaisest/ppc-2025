#include <cmath>
#include <memory>
#include <vector>
#include <iostream>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
constexpr int LANES = 4;
typedef double f64x4 __attribute__((vector_size(LANES * sizeof(double))));
void correlate(const int ny, const int nx, const float *data, float *result)
{
    std::vector<f64x4> norm;
    // if nx is not divisible by LANES, there's going to be an additional overflow block
    const int row_blocks = nx / LANES + (nx % LANES != 0 ? 1 : 0);
    const int rem = nx % LANES;
    norm.reserve(row_blocks * ny);
    // approach as suggested in the tip
    for (int i = 0; i < ny; i++)
    {
        double sum = 0;
        f64x4 arr = {};

        int counter = 0;
        for (int x = 0; x < nx; x++)
        {
            arr[counter] = data[x + i * nx];
            sum += data[x + i * nx];
            counter++;
            if (counter == LANES)
            {
                norm.push_back(arr);
                for (int k = 0; k < LANES; k++)
                {
                    arr[k] = 0;
                }

                counter = 0;
            }
        }
        if (counter != 0)
        {
            norm.push_back(arr);
        }
        std::vector<f64x4>::iterator start_it = norm.end();
        start_it -= row_blocks;
        // iterator now points at first block that was added in this row

        double mean = sum / (double)nx;
        double sum_sq = 0;
        for (auto it = start_it; it != norm.end(); it++)
        {
            // edge case: don't subtract mean from padded zeros at the end so they stay at zero
            if (it == norm.end() - 1 && rem > 0)
            {
                for (int k = 0; k < rem; k++)
                {
                    (*it)[k] -= mean;
                }
            }
            else
            {
                *it = *it - mean;
            }
            f64x4 square = (*it) * (*it);
            for (int k = 0; k < LANES; k++)
            {
                sum_sq += square[k];
            }
        }

        double sq_sqrt = sqrt(sum_sq);
        for (auto it = start_it; it != norm.end(); it++)
        {
            *it = *it / sq_sqrt;
        }
    }

    for (int x = 0; x < ny; x++)
    {
        for (int y = 0; y <= x; y++)
        {
            f64x4 sum = {};
            for (int k = 0; k < row_blocks; k++)
            {
                const f64x4 lhs = norm[y * row_blocks + k];
                const f64x4 rhs = norm[x * row_blocks + k];
                sum += lhs * rhs;
            }
            double res = 0;
            for (int v = 0; v < LANES; v++)
            {
                res += sum[v];
            }

            result[y * ny + x] = (float)res;
        }
    }
}