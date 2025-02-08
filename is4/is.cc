#include <limits>
#include <iostream>
#include <cmath>
struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
#include <iostream>
#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};
Result segment(int ny, int nx, const float *data)
{
    double *sums = new double[3 * nx * ny]();
    double *sums_sq = new double[3 * nx * ny]();

    // we sum row-wise, so that we get sums of areas from the top left corner of
    // the image to the specified pixel. we also sum squares of each element for
    // the calculation of the cost of each square
    Timer tmr;

    for (int y = 0; y < ny; y++)
    {
        double sum[3] = {0.0, 0.0, 0.0};
        double sum_sq[3] = {0.0, 0.0, 0.0};
        for (int x = 0; x < nx; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                double pix = data[c + 3 * x + 3 * nx * y];
                sum[c] += pix;
                sum_sq[c] += pow(pix, 2.0);

                double prev = y != 0 ? sums[c + 3 * x + 3 * nx * (y - 1)] : 0.0;
                double prev_sq = y != 0 ? sums_sq[c + 3 * x + 3 * nx * (y - 1)] : 0.0;

                sums[c + 3 * x + 3 * nx * y] = prev + sum[c];
                sums_sq[c + 3 * x + 3 * nx * y] = prev_sq + sum_sq[c];
            }
        }
    }
    double t = tmr.elapsed();
    std::cout << t << std::endl;
    tmr.reset();
    // total sum of pixels in image
    double image_totals[3] = {sums[0 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums[1 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums[2 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)]};
    // same, but of squared values
    double image_sq_totals[3] = {sums_sq[0 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums_sq[1 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums_sq[2 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)]};

    double final_error = std::numeric_limits<double>::max();
    Result final_result;

#pragma omp parallel
    {
        double local_min_error = std::numeric_limits<double>::max();
        Result local_result;
#pragma omp for nowait
        for (int h = 1; h <= ny; h++)
        {
            for (int w = 1; w <= nx; w++)
            {
                int inner_area = h * w;
                int outer_area = nx * ny - inner_area;
                // we must have two segments
                if (w == nx && h == ny)
                {
                    continue;
                }
                for (int y = 0; y <= ny - h; y++)
                {
                    constexpr int LANES = 8;
                    const int rem = nx % LANES;
                    for (int x = 0; x <= nx - w; x++)
                    {
                        // indexes of the corners of the inner area
                        int tl = 3 * (x - 1) + 3 * nx * (y - 1);
                        int tr = 3 * (x + w - 1) + 3 * nx * (y - 1);
                        int bl = 3 * (x - 1) + 3 * nx * (h + y - 1);
                        int br = 3 * (x + w - 1) + 3 * nx * (h + y - 1);

                        double outer_sums[3];
                        double inner_sums[3];
                        double outer_sq_sums[3];
                        double inner_sq_sums[3];
                        double inner_color[3];
                        double outer_color[3];
                        double total_error = 0.0;
                        for (int c = 0; c < 3; c++)
                        {
                            // https://ppc-exercises.cs.aalto.fi/static/exercises/is/hint.png
                            // start with area between top left of image and bottom right of area
                            inner_sums[c] = sums[br + c];
                            inner_sq_sums[c] = sums_sq[br + c];
                            // remove area between top left of image and bottom left of area (i.e. remove outer left of area)
                            if (x != 0)
                            {
                                inner_sums[c] -= sums[bl + c];
                                inner_sq_sums[c] -= sums_sq[bl + c];
                            }
                            // remove area between top left of image and top right of area (i.e. remove above of area)
                            if (y != 0)
                            {
                                inner_sums[c] -= sums[tr + c];
                                inner_sq_sums[c] -= sums_sq[tr + c];
                            }
                            // add back doubly removed section
                            if (x != 0 && y != 0)
                            {
                                inner_sums[c] += sums[tl + c];
                                inner_sq_sums[c] -= sums_sq[tl + c];
                            }

                            // outer area
                            outer_sums[c] = image_totals[c] - inner_sums[c];
                            outer_sq_sums[c] = image_sq_totals[c] - inner_sq_sums[c];

                            // inner colour is average of the innear area
                            inner_color[c] = inner_sums[c] / (double)inner_area;
                            // and similarly outer colour is the average of the image, excluding the inner area
                            outer_color[c] = outer_sums[c] / (double)outer_area;

                            // (x-c)^2 + (y-c)^2 + (z-c)^2 ... can be expanded to
                            // x^2 + y^2 + z^2 ... - 2 * c * (x+y+z ...) + n * c^2,
                            // where c is the averaged colour of the area, x-z are
                            // pixel values, and n is the area size
                            total_error += inner_sq_sums[c] - 2 * inner_color[c] * inner_sums[c] + inner_area * pow(inner_color[c], 2.0);
                            total_error += outer_sq_sums[c] - 2 * outer_color[c] * outer_sums[c] + outer_area * pow(outer_color[c], 2.0);
                        }

                        if (total_error < local_min_error)
                        {
                            local_min_error = total_error;
                            local_result.y0 = y;
                            local_result.x0 = x;
                            local_result.y1 = y + h;
                            local_result.x1 = x + w;
                            std::copy(inner_color, inner_color + 3, local_result.inner);
                            std::copy(outer_color, outer_color + 3, local_result.outer);
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            // compare thread results and save best
            if (local_min_error < final_error)
            {
                final_error = local_min_error;
                final_result = local_result;
            }
        }
    }

    t = tmr.elapsed();
    std::cout << t << std::endl;
    delete[] sums;
    delete[] sums_sq;
    return final_result;
}