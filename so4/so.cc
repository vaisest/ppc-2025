#include <algorithm>
#include <vector>
#include <thread>
#include <iostream>

typedef unsigned long long data_t;

// https://en.wikipedia.org/wiki/Merge_sort#Merge_sort_with_parallel_merging
// void pm_sort(data_t *A, int lo, int hi, std::vector<data_t> &B, int off)
// {
//     auto len = hi - lo + 1;
//     if (len == 1)
//     {
//         B[off] = A[lo];
//     }
//     else
//     {
//         std::vector<data_t> T(len);
//         auto mid = (lo + hi) / 2;
//         auto midp = mid - lo + 1;
//         std::thread left(pm_sort, A, lo, mid, T, 0);
//         pm_sort(A, mid + 1, hi, T, midp + 1);
//     }
// }

void merge(std::vector<data_t> &left, std::vector<data_t> &right, data_t *out)
{
    // merge by copying from either left or right based on ordering, until both
    // are empty
    auto left_iter = left.begin();
    auto right_iter = right.begin();
    // auto out_iter = out.begin();
    while ((left_iter != left.end()) && (right_iter != right.end()))
    {
        if (*left_iter < *right_iter)
        {
            *out = *left_iter;
            left_iter++;
        }
        else
        {

            *out = *right_iter;
            right_iter++;
        }
        out++;
    }
    while ((left_iter != left.end()))
    {
        *out = *left_iter;
        left_iter++;
        out++;
    }
    while ((right_iter != right.end()))
    {
        *out = *right_iter;
        right_iter++;
        out++;
    }
}

void pm_sort(data_t *data, const size_t n, const size_t sort_size)
{
    // we keep splitting the input in half until it's below the optimal sort
    // size, and upon returning we merge it into the output
    auto mid = n / 2;
    if (n < sort_size)
    {
        std::sort(data, data + n);
        return;
    }

    std::vector<data_t> left(data, data + mid);
    std::vector<data_t> right(data + mid, data + n);

    std::thread left_thread(pm_sort, left.data(), left.size(), sort_size);
    pm_sort(right.data(), right.size(), sort_size);
    left_thread.join();

    merge(left, right, data);
}

void psort(int n, data_t *data)
{
    auto thread_count = std::thread::hardware_concurrency();

    std::cout << thread_count << " threads" << std::endl;

    constexpr unsigned int MIN_SIZE = 512;
    auto target_size = std::max(n / (thread_count * 2), MIN_SIZE);

    pm_sort(data, n, target_size);
}
