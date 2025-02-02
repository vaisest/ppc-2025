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
constexpr int LANES = 8;
typedef double f64x8 __attribute__((vector_size(LANES * sizeof(double))));
constexpr f64x8 zero_vec = {};
#include <iostream>
#include <chrono>
#include <iomanip>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        using namespace std::chrono_literals;
        auto time = (clock_::now() - beg_);
        constexpr auto unit = std::chrono::duration_cast<std::chrono::duration<double>>(1s);
        return time / unit;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double> second_;
    std::chrono::time_point<clock_> beg_;
};

double vec_sum(f64x8 v)
{
    return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}

void correlate(const int ny, const int nx, const float *data, float *result)
{
    std::vector<f64x8> norm;
    // if nx is not divisible by LANES, there's going to be an additional overflow block
    int row_blocks = nx / LANES + (nx % LANES != 0 ? 1 : 0);
    row_blocks = row_blocks + (8 - (row_blocks % 8));
    const int rem = nx % LANES;
    const int rows = ny + (8 - (ny % 8));
    norm.reserve(row_blocks * rows);
    // approach as suggested in the tip
    Timer tmr;

    for (int y = 0; y < rows; y++)
    {
        double sum = 0;
        f64x8 arr = zero_vec;

        int counter = 0;
        if (y >= ny)
        {
            for (int i = 0; i < row_blocks; i++)
            {
                norm.push_back(zero_vec);
            }
            continue;
        }
        for (int x = 0; x < row_blocks; x++)
        {
            for (int v = 0; v < 8; v++)
            {
                int source_x = x * 8 + v;
                const double value = source_x >= nx ? 0.0 : data[source_x + y * nx];
                arr[counter] = value;
                sum += value;
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
        }

        if (counter != 0)
        {
            norm.push_back(arr);
        }

        double mean = sum / (double)nx;
        double sum_sq = 0;
        std::cout << norm.size() << std::endl;
        for (int x = 0; x < nx; x++)
        {
            int block_idx = y * row_blocks + x / 8;
            if (nx - x < 8)
            {
                // partially filled block
                for (int i = 0; i < nx - x; i++)
                {
                    std::cout << block_idx << " " << i << std::endl;
                    norm[block_idx][i] -= mean;
                }
            }
            else
            {
                norm[block_idx] -= mean;
            }

            f64x8 square = norm[block_idx] * norm[block_idx];
            sum_sq += vec_sum(square);
        }

        double sq_sqrt = sqrt(sum_sq);
        for (int x = 0; x < nx; x += 8)
        {
            int block_idx = y * row_blocks + x / 8;
            norm[block_idx] /= sq_sqrt;
        }
    }
    for (int i = 0; i < 8; i++)
    {
        std::cout << "row " << i << ":";
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                std::cout << " " << norm[i * row_blocks + j][k];
            }
        }
        std::cout << std::endl;
    }

    double t = tmr.elapsed();
    std::cout << "prep loop took " << std::setprecision(3) << t << " seconds" << std::endl;
    tmr.reset();
    std::cout << rows << " " << row_blocks << " " << norm.size() << std::endl;
#pragma omp parallel for
    for (int x = 0; x < ny; x += 8)
    {
        for (int y = 0; y <= x; y += 8)
        {
            constexpr int N = 8;
            f64x8 stuff[N * N];
            for (int v = 0; v < N * N; v++)
            {
                stuff[v] = zero_vec;
            }
            std::cout << x << " " << y << std::endl;
            for (int k = 0; k < row_blocks; k += N)
            {
                for (int i = 0; i < N; i++)
                {
                    const f64x8 row = norm[y * row_blocks + i + k];
                    for (int v = 0; v < 8; v++)
                    {
                        std::cout << row[v] << " ";
                    }
                    std::cout << std::endl;

                    for (int j = 0; j < N; j++)
                    {
                        std::cout << k << " " << i << " " << j << std::endl;
                        const f64x8 col = norm[x * row_blocks + j + k];
                        for (int v = 0; v < 8; v++)
                        {
                            std::cout << col[v] << " ";
                        }
                        std::cout << std::endl;
                        stuff[i * N + j] += row * col;
                    }
                }
            }
            for (int v = 0; v < 8; v++)
            {
                std::cout << "row " << v << ":";
                for (int w = 0; w < 8; w++)
                {
                    std::cout << "[";
                    for (int k = 0; k < 8; k++)
                    {

                        std::cout << " " << stuff[v * N + w][k];
                    }
                    std::cout << "] ";
                }
                std::cout << std::endl;
            }

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    const int dest_y = (y + i);
                    const int dest_x = (x + j);
                    if (dest_y < ny && dest_x < ny)
                    {
                        result[dest_y * nx + dest_x] = vec_sum(stuff[i * N + j]);
                    }
                }
            }
        }
    }
    double t2 = tmr.elapsed();
    std::cout << "proc loop took " << std::setprecision(3) << t2 << " seconds" << std::endl;
}