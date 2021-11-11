// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Utils/FileIO.h"

#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace xcore {
namespace utils {

LogicalResult writeDataToFile(std::string &filename, std::string &data) {
  auto outputFile = openOutputFile(filename);
  if (!outputFile) {
    llvm::errs() << "Could not open output file: " << filename << "\n";
    return failure();
  }
  outputFile->os() << data;
  outputFile->keep();
  return success();
}

LogicalResult writeFlashImageToFile(std::string &filename,
                                    std::vector<std::vector<char>> tensorsVec) {

  // Combine data for the tensors
  std::string data;
  for (auto const &tensor : tensorsVec) {
    data += std::string(tensor.data(), tensor.size());
  }

  return utils::writeDataToFile(filename, data);
}

LogicalResult writeMLIRToFlatBufferFile(std::string &filename,
                                        mlir::ModuleOp module) {
  std::string serialized_flatbuffer;
  tflite::FlatbufferExportOptions options;
  bool emit_builtin_tflite_ops = true;
  bool emit_select_tf_ops = true;
  bool emit_custom_ops = true;

  options.toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.toco_flags.set_allow_custom_ops(emit_custom_ops);

  if (tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                &serialized_flatbuffer)) {
    return writeDataToFile(filename, serialized_flatbuffer);
  } else {
    llvm::errs() << "Error converting MLIR to flatbuffer, no file written"
                 << "\n";
    return failure();
  }
}

mlir::OwningModuleRef readFlatBufferFileToMLIR(std::string &filename,
                                               mlir::MLIRContext *context) {
  std::string errorMessage;
  auto inputFile = openInputFile(filename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return mlir::OwningModuleRef(nullptr);
  }

  auto loc = mlir::FileLineColLoc::get(context, filename, 0, 0);
  OwningModuleRef mod =
      tflite::FlatBufferToMlir(absl::string_view(inputFile->getBufferStart(),
                                                 inputFile->getBufferSize()),
                               context, loc);
  return mod;
}

} // namespace utils
} // namespace xcore
} // namespace mlir
