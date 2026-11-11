#include "infer/ThreadPool.h"

#include <algorithm>

namespace infer {

ThreadPool::ThreadPool(int workerCount) {
    if (workerCount < 0) {
        workerCount = 0;
    }
    mWorkers.reserve(workerCount);
    for (int i = 0; i < workerCount; ++i) {
        mWorkers.emplace_back([this, i] { workerLoop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mExit = true;
        ++mGeneration;
    }
    mWake.notify_all();
    for (auto& worker : mWorkers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::runSlices() {
    while (true) {
        const int begin = mNext.fetch_add(mGrain, std::memory_order_relaxed);
        if (begin >= mTotal) {
            return;
        }
        const int end = std::min(begin + mGrain, mTotal);
        (*mTask)(begin, end);
    }
}

void ThreadPool::workerLoop(int) {
    int seen = 0;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mWake.wait(lock, [this, &seen] { return mExit || mGeneration != seen; });
            if (mExit) {
                return;
            }
            seen = mGeneration;
        }
        runSlices();
        {
            std::lock_guard<std::mutex> lock(mMutex);
            ++mFinished;
        }
        mDone.notify_all();
    }
}

void ThreadPool::parallelFor(int total, const std::function<void(int, int)>& fn, int grain) {
    if (total <= 0) {
        return;
    }
    grain = std::max(1, grain);
    if (mWorkers.empty() || total <= grain) {
        fn(0, total);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mTask   = &fn;
        mTotal  = total;
        mGrain  = grain;
        mNext   = 0;
        mFinished = 0;
        ++mGeneration;
    }
    mWake.notify_all();
    runSlices();
    std::unique_lock<std::mutex> lock(mMutex);
    const int workerCount = static_cast<int>(mWorkers.size());
    mDone.wait(lock, [this, workerCount] { return mFinished >= workerCount; });
    mTask = nullptr;
}

} // namespace infer
