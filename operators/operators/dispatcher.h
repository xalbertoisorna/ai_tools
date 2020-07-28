// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_DISPATCHER_H_
#define XCORE_OPERATORS_DISPATCHER_H_

#include <ctime>

#include "operators/device_memory.h"
#include "operators/planning.h"
#include "operators/xcore_reporter.h"
#include "tensorflow/lite/c/common.h"

#ifdef XCORE
extern "C" {
#ifdef _TIME_H_
#define _clock_defined
#endif
#include <xcore/thread.h>
}

#define ATTRIBUTE_THREAD_FUNCTION __attribute__((fptrgroup("thread_function")))

#else  // not XCORE
#include <thread>
#include <vector>

#define ATTRIBUTE_THREAD_FUNCTION

typedef void (*thread_function_t)(void *);
typedef std::vector<std::thread> threadgroup_t;
#endif

namespace xcore {

constexpr size_t kMaxThreads = 5;
constexpr size_t kBytesPerStackword = 4;
constexpr size_t kWordAlignment = 4;
constexpr size_t kDoubleWordAlignment = 8;

typedef struct TaskArray {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  size_t stack_size;
  char *stack;
  int size;
  void *arguments[kMaxThreads];
} TaskArray;

class Dispatcher {
 public:
  Dispatcher(tflite::ErrorReporter *reporter, bool use_current_core = true);
  ~Dispatcher();

  TfLiteStatus InitializeTasks(thread_function_t function, char *stack,
                               size_t stack_size);
  TfLiteStatus AddTask(void *argument);
  TfLiteStatus JoinTasks();

  TfLiteStatus Reset();

  tflite::ErrorReporter *GetReporter();

  void FetchBuffer(int8_t **dest, int8_t const *src, size_t size);
  void FetchWeights(int8_t **dest, int8_t const *src, size_t size,
                    ChannelGroup const &changrp);
  void FetchBiases(int16_t **dest, int16_t const *src, size_t size,
                   ChannelGroup const &changrp);

 private:
  bool use_current_thread_;
  threadgroup_t group_;
  TaskArray tasks_;
  tflite::ErrorReporter *reporter_;
};

// static, shared Dispatcher object
Dispatcher *GetDispatcher();
void SetDispatcher(Dispatcher *);

}  // namespace xcore

#endif  // XCORE_OPERATORS_DISPATCHER_H_