// Replace TFL Transpose with Transpose for XCore.
#include "IR/XCoreOps.h"
#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {

// Replace TFL Transpose with xcore Transpose for XCore.
struct ReplaceTranspose
    : public PassWrapper<ReplaceTranspose, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceTranspose)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect, xcore::XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-transpose"; }
  StringRef getDescription() const final {
    return "Replace TFL Transpose with xcore Transpose for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceTransposePattern : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    // Check that input is a RankedTensorType with static shape
    auto inputType =
        transposeOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape())
      return failure();

    // Get the input shape
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    // Get the permutation operand (second input)
    Value permValue = transposeOp.getPerm();

    // Check that permValue is a constant
    DenseIntElementsAttr permAttr;
    if (!matchPattern(permValue, m_Constant(&permAttr))) {
      return failure();
    }

    // Extract permutation as an array of integers
    SmallVector<int32_t, 4> perm;
    for (auto val : permAttr.getValues<APInt>()) {
      perm.push_back(val.getSExtValue());
    }

    // Compute offsets and t_shape as per the Python code
    // Compute offsets = tuple(np.prod(SHAPE[i+1:], dtype=np.int32) for i in
    // range(len(SHAPE)))
    SmallVector<int32_t, 4> offsets(rank);
    for (int i = 0; i < rank; ++i) {
      int32_t prod = 1;
      for (int j = i + 1; j < rank; ++j) {
        prod *= inputShape[j];
      }
      offsets[i] = prod;
    }

    // Rearrange offsets according to permutation
    SmallVector<int32_t, 4> permutedOffsets(rank);
    for (int i = 0; i < rank; ++i) {
      permutedOffsets[i] = offsets[perm[i]];
    }

    // Compute t_shape = tuple(SHAPE[p] for p in PERM)
    SmallVector<int64_t, 4> tShape(rank);
    for (int i = 0; i < rank; ++i) {
      tShape[i] = inputShape[perm[i]];
    }

    // Create the xcore::TransposeOp with attributes offsets and t_shape
    // Get the output type
    auto outputType = transposeOp.getOutput().getType();

    // Create the new TransposeOp
    auto newTransposeOp = rewriter.create<TransposeOp>(
        transposeOp.getLoc(), outputType, transposeOp.getInput(),
        rewriter.getI32ArrayAttr(permutedOffsets),
        rewriter.getI64ArrayAttr(tShape));

    // Replace the old TransposeOp
    rewriter.replaceOp(transposeOp, newTransposeOp.getResult());

    return success();
  }
};

void ReplaceTranspose::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceTransposePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

// Creates an instance of the ReplaceTranspose pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceTransposePass() {
  return std::make_unique<ReplaceTranspose>();
}

static PassRegistration<ReplaceTranspose> pass;

} // namespace mlir::xcore
