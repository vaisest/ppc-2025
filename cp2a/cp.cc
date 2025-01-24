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
void correlate(int ny, int nx, const float *data, float *result)
{
    for (int i = 0; i < ny; i++)
    {
        constexpr int LANES = 4;
        int rem = i % LANES;

        for (int j = 0; (i >= LANES) && j < i - rem; j += LANES)

        {
            double vi_sum[LANES] = {0, 0, 0, 0};
            double vi_sum2[LANES] = {0, 0, 0, 0};
            double vj_sum[LANES] = {0, 0, 0, 0};
            double vj_sum2[LANES] = {0, 0, 0, 0};
            double vij_sum[LANES] = {0, 0, 0, 0};
            // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Mathematical_properties
            // "convenient single-pass algorithm"
            for (int x = 0; x < nx; x++)
            {
                double iv = data[x + i * nx];
                for (int v = 0; v < LANES; v++)
                {
                    double jv = data[x + (j + v) * nx];
                    vi_sum[v] += iv;
                    vi_sum2[v] += iv * iv;
                    vj_sum[v] += jv;
                    vj_sum2[v] += jv * jv;
                    vij_sum[v] += iv * jv;
                }
            }
            for (int v = 0; v < LANES; v++)
            {
                double corr = (nx * vij_sum[v] - vi_sum[v] * vj_sum[v]) / (sqrt(nx * vi_sum2[v] - vi_sum[v] * vi_sum[v]) * sqrt(nx * vj_sum2[v] - vj_sum[v] * vj_sum[v]));

                result[i + (j + v) * ny] = corr;
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
