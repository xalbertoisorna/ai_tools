// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_POOLING_OPERATORS_H_
#define XCORE_POOLING_OPERATORS_H_

#include <cstdint>

#include "operators/planning.h"
#include "tensorflow/lite/c/common.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace pooling {

struct PoolingParams {
  int32_t pool_h;
  int32_t pool_w;
  int32_t stride_h;
  int32_t stride_w;
};

class MaxPool {
 public:
  MaxPool();
  ~MaxPool() {}

  void Init(TfLiteContext* ctx);
  TfLiteStatus Prepare(TfLiteContext* ctx, int32_t X_h, int32_t X_w,
                       int32_t C_in, int32_t Y_h, int32_t Y_w, int32_t C_out);
  TfLiteStatus Eval(TfLiteContext* ctx, int8_t* Y, const int8_t* X);

  PoolingParams params;
  ExecutionPlan execution_plan;

 private:
  nn_maxpool2d_plan_t plan_;
  nn_pool2d_job_t* jobs_;
  int stack_scratch_index_;
  size_t stack_size_;
};

class AvgPool {
 public:
  AvgPool();
  ~AvgPool() {}

  void Init(TfLiteContext* ctx);
  TfLiteStatus Prepare(TfLiteContext* ctx, int32_t X_h, int32_t X_w,
                       int32_t C_in, int32_t Y_h, int32_t Y_w, int32_t C_out);
  TfLiteStatus Eval(TfLiteContext* ctx, int8_t* Y, const int8_t* X);

  PoolingParams params;
  ExecutionPlan execution_plan;

 private:
  nn_avgpool2d_plan_t plan_;
  nn_pool2d_job_t* jobs_;
  int stack_scratch_index_;
  size_t stack_size_;
};

class AvgPool_Global {
 public:
  AvgPool_Global();
  ~AvgPool_Global() {}

  void Init(TfLiteContext* ctx);
  TfLiteStatus Prepare(TfLiteContext* ctx, int32_t X_h, int32_t X_w,
                       int32_t C_in, int32_t bias, int32_t shift,
                       int32_t scale);
  TfLiteStatus Eval(TfLiteContext* ctx, int8_t* Y, const int8_t* X, int32_t X_h,
                    int32_t X_w, uint32_t C_in);

  ExecutionPlan execution_plan;

 private:
  int32_t bias_;
  nn_avgpool2d_global_plan_t plan_;
  nn_avgpool2d_global_job_t* jobs_;
  int stack_scratch_index_;
  size_t stack_size_;
};

}  // namespace pooling
}  // namespace xcore

#endif  // XCORE_POOLING_OPERATORS_H_