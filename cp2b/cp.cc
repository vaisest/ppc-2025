#include <cmath>
#include <memory>
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
#pragma omp parallel for schedule(static, 2)
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
#pragma omp parallel for schedule(static, 2)
    for (int x = 0; x < ny; x++)
    {
        for (int y = 0; y <= x; y++)
        {
            // must be stored separately due to rounding errors
            double sum = 0;
            for (int k = 0; k < nx; k++)
            {
                sum += norm[y * nx + k] * norm[x * nx + k];
            }
            result[y * ny + x] = (float)sum;
        }
    }
}