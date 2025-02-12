#include <algorithm>
#include <vector>
#include <thread>
#include <iostream>

typedef unsigned long long data_t;

void pm_sort(data_t *data, const size_t n, const size_t sort_threshold)
{
    // we keep splitting the input in half until it's below the optimal sort
    // size, and upon returning we merge it into the output
    auto mid = n / 2;
    if (n < sort_threshold)
    {
        std::sort(data, data + n);
        return;
    }

    std::vector<data_t> left(data, data + mid);
    std::vector<data_t> right(data + mid, data + n);

    std::thread left_thread(pm_sort, left.data(), left.size(), sort_threshold);

    pm_sort(right.data(), right.size(), sort_threshold);

    left_thread.join();

    std::merge(left.begin(), left.end(), right.begin(), right.end(), data);
}

void psort(int n, data_t *data)
{
    auto thread_count = std::thread::hardware_concurrency();
    // "If the value is not well defined or not computable, returns ​0​."
    if (thread_count == 0)
    {
        thread_count = 32;
    }

    constexpr unsigned int MIN_SIZE = 32;
    // we aim to split into a size where every core gets to sort a part of the array
    auto target_size = std::max(n / (thread_count), MIN_SIZE);

    pm_sort(data, n, target_size);
}
