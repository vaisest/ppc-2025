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
Result segment(int ny, int nx, const float *data)
{
    double *sums = new double[3 * nx * ny]();
    // we sum row-wise, so that we get sums of areas from the top left corner of
    // the image to the specified pixel
    for (int y = 0; y < ny; y++)
    {
        double sum[3] = {0.0, 0.0, 0.0};
        for (int x = 0; x < nx; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                sum[c] += data[c + 3 * x + 3 * nx * y];

                double prev = y != 0 ? sums[c + 3 * x + 3 * nx * (y - 1)] : 0.0;

                sums[c + 3 * x + 3 * nx * y] = prev + sum[c];
            }
        }
    }
    // total sum of pixels in image
    double image_totals[3] = {sums[0 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums[1 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)], sums[2 + 3 * (nx - 1) + 3 * (nx) * (ny - 1)]};
    double min_error = std::numeric_limits<double>::max();
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    for (int h = 1; h <= ny; h++)
    {
        for (int w = 1; w <= nx; w++)
        {
            int area_size = h * w;
            // we must have two segments
            if (w == nx && h == ny)
            {
                continue;
            }
            for (int y = 0; y <= ny - h; y++)
            {
                for (int x = 0; x <= nx - w; x++)
                {
                    int tl = 3 * (x - 1) + 3 * nx * (y - 1);
                    int tr = 3 * (x + w - 1) + 3 * nx * (y - 1);
                    int bl = 3 * (x - 1) + 3 * nx * (h + y - 1);
                    int br = 3 * (x + w - 1) + 3 * nx * (h + y - 1);

                    double area_sums[3];
                    for (int c = 0; c < 3; c++)
                    {
                        // https://ppc-exercises.cs.aalto.fi/static/exercises/is/hint.png
                        // start with area between top left of image and bottom right of area
                        area_sums[c] = sums[br + c];
                        // remove area between top left of image and bottom left of area (i.e. remove outer left of area)
                        if (x != 0)
                        {
                            area_sums[c] -= sums[bl + c];
                        }
                        // remove area between top left of image and top right of area (i.e. remove above of area)
                        if (y != 0)
                        {
                            area_sums[c] -= sums[tr + c];
                        }
                        // add back doubly removed section
                        if (x != 0 && y != 0)
                        {
                            area_sums[c] += sums[tl + c];
                        }
                    }
                    // inner colour is average of the innear area
                    double inner_color[3] = {area_sums[0] / (double)area_size, area_sums[1] / (double)area_size, area_sums[2] / (double)area_size};
                    // and similarly outer colour is the average of the image, excluding the inner area
                    double outer_color[3] = {(image_totals[0] - area_sums[0]) / (double)(nx * ny - area_size), (image_totals[1] - area_sums[1]) / (double)(nx * ny - area_size), (image_totals[2] - area_sums[2]) / (double)(nx * ny - area_size)};
                    double total_se = 0.0;
                    for (int y2 = 0; y2 < ny; y2++)
                    {
                        for (int x2 = 0; x2 < nx; x2++)
                        {
                            for (int c = 0; c < 3; c++)
                            {
                                if (x2 >= x && x2 < x + w && y2 >= y && y2 < y + h)
                                {
                                    total_se += pow(inner_color[0] - data[0 + 3 * x2 + 3 * nx * y2], 2.0);
                                    total_se += pow(inner_color[1] - data[1 + 3 * x2 + 3 * nx * y2], 2.0);
                                    total_se += pow(inner_color[2] - data[2 + 3 * x2 + 3 * nx * y2], 2.0);
                                }
                                else
                                {
                                    total_se += pow(outer_color[0] - data[0 + 3 * x2 + 3 * nx * y2], 2.0);
                                    total_se += pow(outer_color[1] - data[1 + 3 * x2 + 3 * nx * y2], 2.0);
                                    total_se += pow(outer_color[2] - data[2 + 3 * x2 + 3 * nx * y2], 2.0);
                                }
                            }
                        }
                    }
                    if (total_se < min_error)
                    {
                        min_error = total_se;
                        result.y0 = y;
                        result.x0 = x;
                        result.y1 = y + h;
                        result.x1 = x + w;
                        result.inner[0] = (float)inner_color[0];
                        result.inner[1] = (float)inner_color[1];
                        result.inner[2] = (float)inner_color[2];
                        result.outer[0] = (float)outer_color[0];
                        result.outer[1] = (float)outer_color[1];
                        result.outer[2] = (float)outer_color[2];
                    }
                }
            }
        }
    }
    delete[] sums;
    return result;
}
