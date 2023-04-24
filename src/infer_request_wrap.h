// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <queue>
#include <string>
#include <vector>

// clang-format off
// clang-format on

typedef std::function<void(size_t id)> QueueCallbackFunction;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks and calculates execution
/// time.
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(ov::CompiledModel& model, size_t id, QueueCallbackFunction callbackQueue)
        : _request(model.create_infer_request()),
          _id(id),
          _callbackQueue(callbackQueue) {
        _request.set_callback([&](const std::exception_ptr& ptr) {
        _callbackQueue(_id);
        });
    }

    void start_async() {
        _request.start_async();
    }

    void wait() {
        _request.wait();
    }

    void infer() {
        _request.infer();
        _callbackQueue(_id);
    }

    std::vector<ov::ProfilingInfo> get_performance_counts() {
        return _request.get_profiling_info();
    }

    void set_shape(const std::string& name, const ov::Shape& dims) {
        // TODO check return status
        _request.get_tensor(name).set_shape(dims);
    }

    ov::Tensor get_tensor(const std::string& name) {
        return _request.get_tensor(name);
    }

    void set_tensor(const std::string& name, const ov::Tensor& data) {
        _request.set_tensor(name, data);
    }

    ov::InferRequest _request;
    size_t _id;

private:
    QueueCallbackFunction _callbackQueue;
};

class InferRequestsQueue final {
public:
    InferRequestsQueue(ov::CompiledModel& model, size_t nireq){
        for (size_t id = 0; id < nireq; id++) {
            requests.push_back(std::make_shared<InferReqWrap>(model,
                                                              id,
                                                              std::bind(&InferRequestsQueue::put_idle_request,
                                                                        this,
                                                                        std::placeholders::_1
                                                                        )));
            _idleIds.push(id);
        }
    }

    ~InferRequestsQueue() {
        // Inference Request guarantee that it will wait for all asynchronous internal tasks in destructor
        // So it should be released before any context that the request can use inside internal asynchronous tasks
        // For example all members of InferRequestsQueue would be destroyed before `requests` vector
        // So requests can try to use this members from `putIdleRequest()` that would be called from request callback
        // To avoid this we should move this vector declaration after all members declaration or just clear it manually
        // in destructor
        requests.clear();
    }

    void put_idle_request(size_t id) {
        std::unique_lock<std::mutex> lock(_mutex);
        _idleIds.push(id);
        //printf("put_idle_request: %ld\n", id);
        _cv.notify_one();
    }

    InferReqWrap::Ptr get_idle_request() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return _idleIds.size() > 0;
        });
        auto request = requests.at(_idleIds.front());
        _idleIds.pop();
        //printf("get_idle_request: %ld\n", _idleIds.front());
        return request;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            //printf("requests.size(): %ld\n", requests.size());
            //printf("_idleIds.size(): %ld\n", _idleIds.size());
            return _idleIds.size() == requests.size();
        });
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> _idleIds;
    std::mutex _mutex;
    std::condition_variable _cv;
};
