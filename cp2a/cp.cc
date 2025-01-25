#include <cmath>
#include <memory>
#include <iostream>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(const int ny, const int nx, const float *data, float *result)
{
    std::unique_ptr<double[]> norm(new double[ny * nx]);
    // approach as suggested in the tip
    for (int i = 0; i < ny; i++)
    {
        std::copy(data + nx * i, data + nx * (i + 1), norm.get() + nx * i);

        double sum = 0;
        for (int x = 0; x < nx; x++)
        {
            sum += norm[x + i * nx];
        }

        double mean = sum / (double)nx;
        double sum_sq = 0;
        for (int x = 0; x < nx; x++)
        {
            double asd = norm[x + i * nx] - mean;
            norm[x + i * nx] = asd;
            sum_sq += asd * asd;
        }
        double sq_sqrt = sqrt(sum_sq);
        for (int x = 0; x < nx; x++)
        {
            norm[x + i * nx] /= sq_sqrt;
        }
    }
    for (int x = 0; x < ny; x++)
    {
        for (int y = 0; y <= x; y++)
        {
            // 6 seems to be optimal here. might depend on the processor?
            constexpr int LANES = 6;
            // must be stored separately due to rounding errors as the destination is float
            double sums[LANES] = {};
            const int rem = nx % LANES;
            const int end = nx - rem;
            for (int k = 0; k < end; k += LANES)
            {
                for (int i = 0; i < LANES; i++)
                {
                    sums[i] += norm[y * nx + k + i] * norm[x * nx + k + i];
                }
            }
            double res = 0;
            for (int k = end; k < nx; k++)
            {
                res += norm[y * nx + k] * norm[x * nx + k];
            }
            for (int i = 0; i < LANES; i++)
            {
                res += sums[i];
            }
            result[y * ny + x] = (float)res;
        }
    }
}