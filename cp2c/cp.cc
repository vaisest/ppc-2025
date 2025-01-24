/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <iostream>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));
double4_t sqrtv(double4_t nums)
{
    // error: cannot convert ‘<brace-enclosed initializer list>’ to ‘double4_t’ {aka ‘__vector(4) double’} in return
    double4_t res = {sqrt(nums[0]),
                     sqrt(nums[1]),
                     sqrt(nums[2]),
                     sqrt(nums[3])};
    return res;
}

void correlate(int ny, int nx, const float *data, float *result)
{
    for (int i = 0; i < ny; i++)
    {
        constexpr int LANES = 4;
        int rem = i % LANES;
        for (int j = 0; (i >= LANES) && j < i - rem; j += LANES)

        {
            double4_t jv = {0, 0, 0, 0};
            double4_t vi_sum = {0, 0, 0, 0};
            double4_t vi_sum2 = {0, 0, 0, 0};
            double4_t vj_sum = {0, 0, 0, 0};
            double4_t vj_sum2 = {0, 0, 0, 0};
            double4_t vij_sum = {0, 0, 0, 0};
            // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Mathematical_properties
            // "convenient single-pass algorithm"
            for (int x = 0; x < nx; x++)
            {
                double iv = data[x + i * nx];
                for (int v = 0; v < LANES; v++)
                {
                    jv[v] = data[x + (j + v) * nx];
                }
                vi_sum += iv;
                vi_sum2 += iv * iv;
                vj_sum += jv;
                vj_sum2 += jv * jv;
                vij_sum += iv * jv;
            }
            double4_t corr = ((double)nx * vij_sum - vi_sum * vj_sum) / (sqrtv((double)nx * vi_sum2 - vi_sum * vi_sum) * sqrtv((double)nx * vj_sum2 - vj_sum * vj_sum));
            for (int v = 0; v < LANES; v++)
            {
                result[i + (j + v) * ny] = corr[v];
            }
        }

        for (int j = i - rem; j <= i; j++)
        {
            double i_sum = 0;
            double i_sum2 = 0;
            double j_sum = 0;
            double j_sum2 = 0;
            double ij_sum = 0;
            // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Mathematical_properties
            // "convenient single-pass algorithm"
            for (int x = 0; x < nx; x++)
            {
                double iv = data[x + i * nx];
                double jv = data[x + j * nx];
                i_sum += iv;
                i_sum2 += iv * iv;
                j_sum += jv;
                j_sum2 += jv * jv;
                ij_sum += iv * jv;
            }
            double corr = (nx * ij_sum - i_sum * j_sum) / (sqrt(nx * i_sum2 - i_sum * i_sum) * sqrt(nx * j_sum2 - j_sum * j_sum));

            result[i + j * ny] = corr;
        }
    }
}
