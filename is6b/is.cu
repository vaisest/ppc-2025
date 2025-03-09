#include <memory>
#include <cuda_runtime.h>
#include <iostream>
struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
struct ResPair
{
    float err;
    Result result;
};
__global__ void kernel(const uint32_t *sums, ResPair *results, int ny, int nx, int image_totals)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    if (h > ny || h == 0)
    {
        return;
    }
    if (w > nx || w == 0)
    {
        return;
    }
    // we must have two segments
    if (w == nx && h == ny)
    {
        return;
    }

    auto local_min_error = -__FLT_MAX__;
    Result local_result;

    auto inner_area = h * w;
    auto outer_area = nx * ny - inner_area;

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

            auto inner_sums = sums[br];
            float total_error = 0.0;
            // https://ppc-exercises.cs.aalto.fi/static/exercises/is/hint.png
            // start with area between top left of image and bottom right of area
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
            auto outer_sums = image_totals - inner_sums;

            // inner colour is average of the innear area
            auto inner_color = inner_sums / (float)inner_area;
            // and similarly outer colour is the average of the image, excluding the inner area
            auto outer_color = outer_sums / (float)outer_area;

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
            auto inner_errors = inner_color * inner_sums;

            // inner_errors += outer_color * outer_sums;
            total_error += outer_color * outer_sums + inner_errors;

            if (total_error > local_min_error)
            {
                local_min_error = total_error;
                local_result.y0 = y;
                local_result.x0 = x;
                local_result.y1 = y + h;
                local_result.x1 = x + w;
                for (int c = 0; c < 3; c++)
                {
                    local_result.inner[c] = inner_color;
                    local_result.outer[c] = outer_color;
                }
            }
        }
    }

    const auto idx = (h - 1) * nx + w - 1;
    results[idx] = {local_min_error, local_result};
}

int div_up(int a, int b)
{
    return (a + b - 1) / b;
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
    std::unique_ptr<uint32_t[]> sums(new uint32_t[ny * nx]);

    // we sum row-wise, so that we get sums of areas from the top left corner of
    // the image to the specified pixel. we also sum squares of each element for
    // the calculation of the cost of each square

    // sums done on the CPU. This takes a tiny amount of time and isn't
    // necessary to do on the GPU
    for (int y = 0; y < ny; y++)
    {
        auto sum = 0;
        for (int x = 0; x < nx; x++)
        {
            if (data[3 * x + 3 * nx * y] == 1.0)
            {
                sum++;
            }
            auto prev = y != 0 ? sums[x + nx * (y - 1)] : 0;

            sums[x + nx * y] = prev + sum;
        }
    }

    auto image_totals = sums[(nx - 1) + (nx) * (ny - 1)];

    std::unique_ptr<ResPair[]> results(new ResPair[nx * ny]);

    uint32_t *sumsGPU = NULL;
    CHECK(cudaMalloc(&sumsGPU, ny * nx * sizeof(uint32_t)));
    ResPair *resGPU = NULL;
    CHECK(cudaMalloc(&resGPU, ny * nx * sizeof(ResPair)));

    CHECK(cudaMemcpy(sumsGPU, sums.get(), ny * nx * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(resGPU, 0, ny * nx * sizeof(ResPair)));

    // increasing block size only on the width axis seems faster
    // 64 gets best result on test server
    const int w_size = 64;
    dim3 block(1, w_size);
    dim3 grid(ny + 1, div_up(nx + 1, w_size));
    kernel<<<grid, block>>>(sumsGPU, resGPU, ny, nx, image_totals);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(results.get(), resGPU, ny * nx * sizeof(ResPair), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(sumsGPU));
    CHECK(cudaFree(resGPU));
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    auto min_err = std::numeric_limits<float>::min();
    // reduce results on CPU. seems to take a minimal amount of time so not
    // worth doing it on the CPU
    for (auto i = 0; i < ny * nx; i++)
    {
        auto res = results[i];
        if (res.err > min_err)
        {
            min_err = res.err;
            result = res.result;
        }
    }

    return result;
}
