#include <limits>
#include <iostream>
#include <immintrin.h>
#include <memory>
struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

typedef double f64x4 __attribute__((vector_size(4 * sizeof(double))));
constexpr f64x4 zero_pix = {};

inline double pix_sum(f64x4 v)
{
    return ((v[0] + v[1]) + (v[2]));
}

inline f64x4 vec_fmadd(f64x4 a, f64x4 b, f64x4 c)
{
    return (a * b) + c;
}
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data)
{
    std::unique_ptr<f64x4[]> sums(new f64x4[ny * nx]);

    // we sum row-wise, so that we get sums of areas from the top left corner of
    // the image to the specified pixel. we also sum squares of each element for
    // the calculation of the cost of each square

    for (int y = 0; y < ny; y++)
    {
        f64x4 sum = zero_pix;
        for (int x = 0; x < nx; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                double pix = data[c + 3 * x + 3 * nx * y];
                sum[c] += pix;
            }
            f64x4 prev = y != 0 ? sums[x + nx * (y - 1)] : zero_pix;

            sums[x + nx * y] = prev + sum;
        }
    }

    // total sum of pixels in image
    f64x4 image_totals = sums[(nx - 1) + (nx) * (ny - 1)];
    // same, but of squared values

    double final_error = std::numeric_limits<double>::min();
    Result final_result;

#pragma omp parallel
    {
        double local_min_error = std::numeric_limits<double>::min();
        Result local_result;
#pragma omp for nowait schedule(static, 1)
        for (int h = 1; h <= ny; h++)
        {
            for (int w = 1; w <= nx; w++)
            {
                double inner_area = h * w;
                double outer_area = nx * ny - inner_area;
                // we must have two segments
                if (w == nx && h == ny)
                {
                    continue;
                }
                for (int y = 0; y <= ny - h; y++)
                {
                    const int top_row = nx * (y - 1);
                    const int bot_row = nx * (y + h - 1);
                    const int end = nx - w;
                    for (int x = 0; x <= end; x++)
                    {
                        // indices of the corners of the inner area
                        const int left = (x - 1);
                        const int right = (x + w - 1);
                        const int tl = top_row + left;
                        const int tr = top_row + right;
                        const int bl = bot_row + left;
                        const int br = bot_row + right;

                        f64x4 outer_sums;
                        f64x4 inner_sums;
                        f64x4 inner_color;
                        f64x4 outer_color;
                        double total_error = {};
                        // https://ppc-exercises.cs.aalto.fi/static/exercises/is/hint.png
                        // start with area between top left of image and bottom right of area
                        inner_sums = sums[br];
                        // remove area between top left of image and bottom left of area (i.e. remove outer left of area)
                        if (x != 0)
                        {
                            inner_sums -= sums[bl];
                        }
                        // remove area between top left of image and top right of area (i.e. remove above of area)
                        if (y != 0)
                        {
                            inner_sums -= sums[tr];
                        }
                        // add back doubly removed section
                        if (x != 0 && y != 0)
                        {
                            inner_sums += sums[tl];
                        }

                        // outer area
                        outer_sums = image_totals - inner_sums;

                        // inner colour is average of the innear area
                        inner_color = inner_sums / (double)inner_area;
                        // and similarly outer colour is the average of the image, excluding the inner area
                        outer_color = outer_sums / (double)outer_area;

                        // from the IS2 solution we can notice that if we expand the
                        // error calculation, some parts of it cancel out. We're
                        // left with `inner_sq_sums - inner_color * inner_sums`.

                        // I'm not sure how or if the math checks out, but it seems
                        // the inner_sq_sums can too be removed without getting a
                        // wrong result.

                        // This results in a lot less calculation needed with
                        // `-inner_color * inner_sums`. Additionally we can remove
                        // the minus sign and flip the comparisons, which means that
                        // as this grows, the error diminishes
                        f64x4 inner_errors = inner_color * inner_sums;

                        // inner_errors += outer_color * outer_sums;
                        total_error += pix_sum(vec_fmadd(outer_color, outer_sums, inner_errors));

                        if (total_error > local_min_error)
                        {
                            local_min_error = total_error;
                            local_result.y0 = y;
                            local_result.x0 = x;
                            local_result.y1 = y + h;
                            local_result.x1 = x + w;
                            for (int c = 0; c < 3; c++)
                            {
                                local_result.inner[c] = inner_color[c];
                                local_result.outer[c] = outer_color[c];
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            // compare thread results and save best
            if (local_min_error > final_error)
            {
                final_error = local_min_error;
                final_result = local_result;
            }
        }
    }

    return final_result;
}
