#include <algorithm>
#include <thread>

typedef unsigned long long data_t;

int hoare_partition(data_t *data, int low, int high)
{
    const auto mid = (low + high) / 2;
    // // select median of low, mid, and high as pivot
    auto pivot = std::max(std::min(data[low], data[mid]), std::min(std::max(data[low], data[mid]), data[high]));

    auto i = low - 1;
    auto j = high + 1;

    while (true)
    {
        i++;
        j--;
        while (data[i] < pivot)
        {
            i++;
        }
        while (data[j] > pivot)
        {
            j--;
        }
        if (i >= j)
        {
            return j;
        }
        std::swap(data[i], data[j]);
    }
}

void qpsort2(data_t *data, int low, int high, const int target_size)
{
    while (true)
    {
        if (low >= high || low < 0)
        {
            return;
        }
        const auto size = high - low;
        if (size <= target_size)
        {
            std::sort(data + low, data + high + 1);
            return;
        }
        // possible to shortcut here to std::sort
        auto p = hoare_partition(data, low, high);
#pragma omp task
        qpsort2(data, low, p, target_size);
        low = p + 1;
    }
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
    auto target_size = std::max(n / (thread_count * 4), MIN_SIZE);
#pragma omp parallel
#pragma omp single
    {
        qpsort2(data, 0, n - 1, target_size);
    }
}
