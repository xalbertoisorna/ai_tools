#include "IR/XCoreOps.h"
#include "Utils/Util.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir::xcore {

namespace {
// Replace TFL Mean with Mean or Mean16 for XCore.
struct ReplaceMean
    : public PassWrapper<ReplaceMean, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMean)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-mean"; }
  StringRef getDescription() const final {
    return "Replace TFL Mean with Mean or Mean16 for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceMeanPattern : public OpRewritePattern<TFL::MeanOp> {
  using OpRewritePattern<TFL::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MeanOp meanOp,
                                PatternRewriter &rewriter) const override {

    auto input = meanOp.getInput();
    auto output = meanOp.getOutput();

    DenseElementsAttr axisAttr;
    if (!matchPattern(meanOp.getAxis(), m_Constant(&axisAttr))) {
      return failure();
    }
    auto axisValues = axisAttr.getValues<int32_t>();
    std::vector<int32_t> axis(axisValues.begin(), axisValues.end());
    int32_t minAxis = *std::min_element(axis.begin(), axis.end());
    int32_t maxAxis = *std::max_element(axis.begin(), axis.end());
    if (maxAxis - minAxis > axis.size() - 1) {
      return failure();
    }

    auto inputType = input.getType().cast<ShapedType>();
    auto outputType = output.getType().cast<ShapedType>();

    // Check if input and output are either int8 or int16.
    bool isInt8 = utils::isNBitSignedQType<8>(inputType.getElementType()) &&
                  utils::isNBitSignedQType<8>(outputType.getElementType());

    bool isInt16 = utils::isNBitSignedQType<16>(inputType.getElementType()) &&
                   utils::isNBitSignedQType<16>(outputType.getElementType());

    if (!(isInt8 || isInt16)) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    int rank = inputShape.size();

    int beginDims = 1;
    for (int i = 0; i < minAxis; i++) {
      beginDims *= inputShape[i];
    }

    int endDims = 1;
    for (int i = maxAxis + 1; i < rank; i++) {
      endDims *= inputShape[i];
    }

    int meanDims = 1;
    for (int i = minAxis; i <= maxAxis; i++) {
      meanDims *= inputShape[i];
    }

    auto inputQType = utils::getQType(input);
    auto outputQType = utils::getQType(output);

    float scaleMul =
        inputQType.getScale() / outputQType.getScale() / static_cast<float>(meanDims);
    auto scaleMulAttr = rewriter.getF32FloatAttr(scaleMul);

    auto beginDimsAttr = rewriter.getI32IntegerAttr(beginDims);
    auto endDimsAttr = rewriter.getI32IntegerAttr(endDims);
    auto meanDimsAttr = rewriter.getI32IntegerAttr(meanDims);

    if (isInt8) {
      float inZeroPoint = static_cast<float>(inputQType.getZeroPoint());
      float outZeroPoint = static_cast<float>(outputQType.getZeroPoint());
      auto inZeroPointAttr = rewriter.getF32FloatAttr(inZeroPoint);
      auto outZeroPointAttr = rewriter.getF32FloatAttr(outZeroPoint);

      auto xcMeanOp = rewriter.create<MeanOp>(
          meanOp.getLoc(), meanOp.getType(), meanOp.getInput(), beginDimsAttr,
          meanDimsAttr, endDimsAttr, inZeroPointAttr, outZeroPointAttr,
          scaleMulAttr);
      rewriter.replaceOp(meanOp, xcMeanOp.getOutput());
    } else { // isInt16
      // Zero points are always zero for int16 and are not passed to Mean16Op.
      auto xcMeanOp = rewriter.create<Mean16Op>(
          meanOp.getLoc(), meanOp.getType(), meanOp.getInput(), beginDimsAttr,
          meanDimsAttr, endDimsAttr, scaleMulAttr);
      rewriter.replaceOp(meanOp, xcMeanOp.getOutput());
    }

    return success();
  }
};

void ReplaceMean::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceMeanPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceMean pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMeanPass() {
  return std::make_unique<ReplaceMean>();
}

static PassRegistration<ReplaceMean> pass;

} // namespace mlir::xcore
