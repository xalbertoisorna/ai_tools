// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_PASSES_H
#define XFORMER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace xcore {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed passes in the right
// order.
void buildXCorePassPipeline(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// XCore-specific passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createApplyPatternsPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceAvgPoolWithConv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceFCWithConv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createPad3to4Conv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceWithConv2DV2Pass();
std::unique_ptr<OperationPass<FuncOp>> createApplyPatterns2Pass();
// std::unique_ptr<OperationPass<FuncOp>> createLegalizeFullyConnectedPass();
std::unique_ptr<OperationPass<FuncOp>> createTranslateToCustomOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerXCorePassPipeline();

inline void registerAllPasses() {
  registerXCorePassPipeline();

  createApplyPatternsPass();
  createReplaceAvgPoolWithConv2DPass();
  createReplaceFCWithConv2DPass();
  createPad3to4Conv2DPass();
  createReplaceWithConv2DV2Pass();
  createApplyPatterns2Pass();
  // createLegalizeFullyConnectedPass();
  createTranslateToCustomOpPass();
}

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_PASSES_H
