#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace infer {

// A persistent thread pool used to parallelize intra-operator work.
// The calling thread participates in every parallel region, so a pool created
// with N workers uses N+1 threads in total.
class ThreadPool {
public:
    explicit ThreadPool(int workerCount);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Total number of participating threads (workers + caller).
    int threadNumber() const {
        return static_cast<int>(mWorkers.size()) + 1;
    }

    // Split [0, total) into chunks of at least `grain` elements across threads.
    void parallelFor(int total, const std::function<void(int begin, int end)>& fn, int grain = 1);

private:
    void runSlices();
    void workerLoop(int workerIndex);

    std::vector<std::thread> mWorkers;
    std::mutex mMutex;
    std::condition_variable mWake;
    std::condition_variable mDone;

    const std::function<void(int, int)>* mTask = nullptr;
    int mTotal                                 = 0;
    int mGrain                                 = 1;
    std::atomic<int> mNext{0};
    int mFinished = 0;
    int mGeneration = 0;
    bool mExit      = false;
};

} // namespace infer
