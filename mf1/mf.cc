/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <algorithm>
#include <utility>
#include <vector>
void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
  float *arr = new float[(hx * 2 + 1) * (hy * 2 + 1)];
  for (int y = 0; y < ny; y++)
  {
    const int ystart = std::max(y - hy, 0);
    const int yend = std::min((hy + y + 1), ny);
    const int rows = yend - ystart;
    for (int x = 0; x < nx; x++)
    {
      const int xstart = std::max(x - hx, 0);
      const int xend = std::min((hx + x + 1), nx);
      const int cols = xend - xstart;

      const int size = rows * cols;

      for (int b = 0; b < rows; b++)
      {
        const int in_row_idx = (ystart + b) * nx;
        const int arr_row_idx = cols * b;
        for (int a = 0; a < cols; a++)
        {
          arr[a + arr_row_idx] = in[xstart + a + in_row_idx];
        }
      }

      const auto mid = arr + size / 2;
      const auto rhs = arr + size / 2 - 1;
      std::nth_element(arr, mid, arr + size);
      out[x + nx * y] = *mid;
      if (size % 2 == 0)
      {
        // save this so the second nth_element doesn't scramble it

        std::nth_element(arr, rhs, arr + size);

        out[x + nx * y] += *rhs;
        out[x + nx * y] /= 2.0;
      }
    }
  }
  delete[] arr;
}
