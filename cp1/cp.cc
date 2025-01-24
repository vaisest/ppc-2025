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

// A reasonable way to calculate all pairwise correlations is the following:

// First normalize the input rows so that each row has the arithmetic mean of 0 — be careful to do the normalization so that you do not change pairwise correlations.
// Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.
// Let X be the normalized input matrix.
// Calculate the (upper triangle of the) matrix product Y = XX^T.
// Now matrix Y contains all pairwise correlations. The only computationally-intensive part is the computation of the matrix product;
// the normalizations can be done in linear time in the input size.
void correlate(int ny, int nx, const float *data, float *result)
{
    double *norm = new double[ny * nx]();
    double *row = new double[nx]();
    for (int i = 0; i < ny; i++)
    {
        double sum = 0;
        double sum_sq = 0;
        for (int x = 0; x < nx; x++)
        {
            float elem = data[x + i * nx];
            row[x] = elem;
            sum += elem;
            sum_sq += elem * elem;
        }
        double mean = sum / (double)nx;
        std::cout << "mean: " << mean << std::endl;
        double sq_sqrt = sqrt(sum_sq);
        std::cout << "sq_sqrt: " << sq_sqrt << std::endl;
        for (int x = 0; x < nx; x++)
        {
            std::cout << "before: " << row[x] << " ";
            // faulty
            row[x] -= mean;
            row[x] /= sq_sqrt;
            std::cout << "after: " << row[x] << std::endl;
        }
        std::copy(row, row + nx, norm + nx * i);
    }
    for (int y = 0; y < ny; y++)
    {
        std::cout << "row:";
        for (int x = 0; x < nx; x++)
        {
            std::cout << " " << norm[x + y * nx];
        }
        std::cout << std::endl;
    }

    double *normt = new double[ny * nx]();
    // transpose
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            normt[i * ny + j] = norm[j * nx + i];
        }
    }

    for (int y = 0; y < nx; y++)
    {
        std::cout << "rowt:";
        for (int x = 0; x < ny; x++)
        {
            std::cout << " " << normt[x + y * ny];
        }
        std::cout << std::endl;
    }

    // result will be yxy
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < ny; x++)
        {
            if (y > x)
            {
                result[y * ny + x] = 6666666.66666;
                continue;
            }
            double sum = 0;
            std::cout << "y: " << y << " x: " << x << std::endl;
            for (int k = 0; k < nx; k++)
            {
                std::cout << "k: " << k << std::endl;
                sum += norm[y * nx + k] * normt[k * ny + x];
            }
            result[y * ny + x] = (float)sum;
        }
    }
    delete[] norm;
    delete[] normt;
    delete[] row;
}
