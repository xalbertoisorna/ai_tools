#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "Utils/ThreadSupport.h"
#include "lib_nn/api/AbstractKernel.hpp"
#include "lib_nn/api/AggregateFn.hpp"
#include "lib_nn/api/MemCpyFn.hpp"
#include "lib_nn/api/OutputTransformFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
struct ReplaceMaxPool2D
    : public PassWrapper<ReplaceMaxPool2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMaxPool2D)
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-maxpool2d"; }
  StringRef getDescription() const final {
    return "Replace TFL MaxPool2D with MaxPool2D for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceMaxPool2DPattern : public OpRewritePattern<TFL::MaxPool2DOp> {
  using OpRewritePattern<TFL::MaxPool2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MaxPool2DOp mPoolOp,
                                PatternRewriter &rewriter) const override {
    auto inputType =
        mPoolOp.getInput().getType().template dyn_cast<RankedTensorType>();
    auto outputType =
        mPoolOp.getOutput().getType().template dyn_cast<RankedTensorType>();
    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);
    auto inputDepth = inputType.getDimSize(3);
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto outputDepth = outputType.getDimSize(3);
    auto splits = utils::getImageRegionThreadSplits(
        threadCountOption, outputHeight, outputWidth, outputDepth);

    auto actualThreadCount = splits.size();
    // Create a string array attr from a vector of strings
    auto getStringArrayAttr = [&](llvm::SmallVector<std::string> value) {
      auto attrs = llvm::to_vector<8>(
          llvm::map_range(value, [&](std::string v) -> Attribute {
            return rewriter.getStringAttr(v);
          }));
      return rewriter.getArrayAttr(attrs);
    };
    int32_t scratchByteParam =
        nn::MatMulInt8::get_scratch_mem_bytes(mPoolOp.getFilterWidth() *
                                              mPoolOp.getFilterHeight()) +
        32; //[asj] FIXME
    nn::ImageGeometry X(inputHeight, inputWidth, inputDepth);
    nn::ImageGeometry Y(outputHeight, outputWidth, outputDepth);
    llvm::SmallVector<std::string> akp;
    for (auto &region : splits) {
      nn::ImageRegion ir(region[0], region[1], region[2], region[3], region[4],
                         region[5]);
      nn::AbstractKernel ak(Y, ir, VPU_INT8_ACC_PERIOD);
      auto akParams = ak.getParams();
      auto akpStr = std::string((char *)&akParams, sizeof(akParams));
      akp.push_back(akpStr);
    }
    nn::ImageRegion ir(0, 0, 0, outputHeight, outputWidth, outputDepth);
    nn::WindowGeometry window(
        mPoolOp.getFilterHeight(), mPoolOp.getFilterWidth(), 1, 0, 0,
        mPoolOp.getStrideH(), mPoolOp.getStrideW(), 1, 1, 1);
    nn::DerefInputFn mf(X, window);
    nn::MatMulDirectFn_DW af(X, window);
    // TODO
    nn::OT_int8_channelwise ot(outputDepth, 0);
    auto mfParams = mf.getParams();
    auto afParams = af.getParams();
    auto otParams = ot.getParams();
    auto mfStr = std::string((char *)&mfParams, sizeof(mfParams));
    auto afStr = std::string((char *)&afParams, sizeof(afParams));
    auto otStr = std::string((char *)&otParams, sizeof(otParams));

    // Create scratch buffer tensor
    Value scratchTensorOp;
    // Create scratch buffer space for all threads
    int32_t scratchBufferSize = scratchByteParam * actualThreadCount;
    if (scratchBufferSize > 0) {
      ShapedType scratchType =
          RankedTensorType::get({static_cast<long long>(scratchBufferSize)},
                                rewriter.getIntegerType(8));
      scratchTensorOp =
          rewriter.create<FakeScratchBufferOp>(mPoolOp.getLoc(), scratchType);
    } else {
      scratchTensorOp = rewriter.create<TFL::NoValueOp>(
          rewriter.getUnknownLoc(), rewriter.getNoneType(),
          rewriter.getUnitAttr());
    }

    auto xcMaxPool2DOp = rewriter.create<MaxPool2DOp>(
        mPoolOp.getLoc(), mPoolOp.getType(), mPoolOp.getInput(),
        scratchTensorOp, rewriter.getStringAttr(mfStr),
        rewriter.getStringAttr(afStr), rewriter.getStringAttr(otStr),
        rewriter.getI32IntegerAttr(scratchByteParam),
        rewriter.getI32IntegerAttr(actualThreadCount), getStringArrayAttr(akp));
    rewriter.replaceOp(mPoolOp, xcMaxPool2DOp.getOutput());
    return success();
  }
};

void ReplaceMaxPool2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceMaxPool2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMaxPoolPass() {
  return std::make_unique<ReplaceMaxPool2D>();
}

static PassRegistration<ReplaceMaxPool2D> pass;

} // namespace mlir::xcore
