
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "lib_ops/api/lib_ops.h"
#include "mobilenet_ops_resolver.h"
#include "mobilenet_v1.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 136200;
uint8_t tensor_arena[kTensorArenaSize];

xcore::Dispatcher *dispatcher = nullptr;
constexpr int num_threads = 5;
constexpr int kXCOREArenaSize = 6000;
uint8_t xcore_arena[kXCOREArenaSize];

static int load_input(const char *filename, char *input, size_t esize) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  if (fsize != esize) {
    printf("Incorrect input file size. Expected %d bytes.\n", esize);
    return 0;
  }

  fseek(fd, 0, SEEK_SET);
  fread(input, 1, esize, fd);
  fclose(fd);

  return 1;
}

static int save_output(const char *filename, const char *output, size_t osize) {
  FILE *fd = fopen(filename, "wb");
  fwrite(output, sizeof(int8_t), osize, fd);
  fclose(fd);

  return 1;
}

static void setup_tflite() {
  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(mobilenet_v1_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Setup xCORE dispatcher (BEFORE calling AllocateTensors)
  static xcore::Dispatcher static_dispatcher(xcore_arena, kXCOREArenaSize,
                                             num_threads);
  xcore::XCoreStatus xcore_status = xcore::InitializeXCore(&static_dispatcher);
  if (xcore_status != xcore::kXCoreOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "InitializeXCore() failed");
    return;
  }
  dispatcher = &static_dispatcher;

  // This pulls in all the operation implementations we need.
  static tflite::ops::micro::xcore::MobileNetOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_tensors_status = interpreter->AllocateTensors();
  if (allocate_tensors_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Two arguments expected: input-file output-file\n");
    return -1;
  }

  char *input_filename = argv[1];
  char *output_filename = argv[2];

  // setup runtime
  setup_tflite();

  // Load input tensor
  if (!load_input(input_filename, input->data.raw, input->bytes)) {
    printf("error loading input filename=%s\n", input_filename);
    return -1;
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
    return -1;
  }

  // save output
  if (!save_output(output_filename, output->data.raw, output->bytes)) {
    printf("error saving output filename=%s\n", output_filename);
    return -1;
  }
  return 0;
}