// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Analysis/MemoryPlan.h"
#include "IR/XCoreOps.h"
#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "xcore-memory-plan"

namespace mlir::xcore {

MemoryPlan::MemoryPlan(Operation *operation)
    : liveness(operation), op(operation) {
  build();
}

void MemoryPlan::build() {
  if (!llvm::isa<func::FuncOp>(op)) {
    return;
  }

  auto funcOp = dyn_cast<func::FuncOp>(op);

  auto getAlignedValueSize = [](Value v) {
    auto type = v.getType().dyn_cast<ShapedType>();
    size_t k = static_cast<size_t>(utils::getShapedTypeSize(type));
    // Align size up to double word = 8 bytes
    k = ((k + 7) / 8) * 8;
    return k;
  };

  for (BlockArgument argument : funcOp.getArguments()) {
    valueInfo.insert(
        {argument,
         {valueInfo.size(), getAlignedValueSize(argument), false, -1, -1}});
    values.push_back(argument);
  }

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == funcOp || llvm::isa<quantfork::StatisticsOp>(op)) {
      return;
    }

    bool isConstantOp = false;
    // TODO(renjieliu): Find a generic way to deal with const ops.
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                  arith::ConstantOp>(op)) {
      isConstantOp = true;
    }

    if (!llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                   arith::ConstantOp>(op)) {
      operationIds.insert({op, operationIds.size()});
      operations.push_back(op);
    }

    for (Value result : op->getResults()) {
      if (result.getType().isa<NoneType>()) {
        continue;
      }
      valueInfo.insert({result,
                        {valueInfo.size(), getAlignedValueSize(result),
                         isConstantOp, -1, -1}});
      values.push_back(result);
    }
  });

  // Liveness
  // Struct with start op and end op
  assert(op->getNumRegions() == 1);
  assert(op->getRegion(0).hasOneBlock());

  Block *block = &op->getRegion(0).front();

  const LivenessBlockInfo *lvb = liveness.getLiveness(block);
  for (auto v : values) {
    Operation *startOp = lvb->getStartOperation(v);
    valueInfo[v].firstUsed = operationIds[startOp];
    valueInfo[v].lastUsed = operationIds[lvb->getEndOperation(v, startOp)];
  }
}

Operation *MemoryPlan::getOpWithMaxMemoryUsed() {
  Block *block = &op->getRegion(0).front();
  const LivenessBlockInfo *lvb = liveness.getLiveness(block);

  int maxSize = -1;
  Operation *maxOp;
  for (auto o : operations) {
    if (o->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::NoValueOp, TFL::QConstOp, TFL::ConstOp,
                  arith::ConstantOp>(o)) {
      continue;
    }
    int size = 0;
    for (auto v : lvb->currentlyLiveValues(o)) {
      if (!valueInfo[v].isConstant)
        size += valueInfo[v].size;
    }
    if (size > maxSize) {
      maxSize = size;
      maxOp = o;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "\nop " << operationIds[o] << " width = " << size);
  }
  LLVM_DEBUG(llvm::dbgs() << "\nMax op " << operationIds[maxOp]
                          << " width = " << maxSize);
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  return maxOp;
}

int MemoryPlan::getOffset(Value v, int size,
                          DenseMap<Value, ValueInfo> &valueInfo,
                          ValuesOrderedByOffset &allocatedValues) {
  int offset = 0;

  // Go through all allocated buffers
  // They are ordered by offset
  for (auto i : allocatedValues) {
    Value allocatedVal = i.first;
    int allocatedOffset = i.second;

    if ((valueInfo[allocatedVal].firstUsed > valueInfo[v].lastUsed) ||
        (valueInfo[v].firstUsed > valueInfo[allocatedVal].lastUsed)) {
      // There is no overlap with this buffer. We move on until we have a clash.
      // When there is a clash, we know we can allocate before that one if there
      // is space as we don't overlap with any of those buffers.
      continue;
    }

    // Found an overlapping buffer
    if (allocatedOffset - offset >= size) {
      // There is a gap
      break;
    } else {
      // Move offset to end of current buffer if larger
      int end = allocatedOffset + valueInfo[allocatedVal].size;
      if (end > offset) {
        offset = end;
      }
    }
  }

  return offset;
}

void MemoryPlan::buildInputOutputTensorMaps(
    llvm::StringMap<Value> &inputTensorMap,
    llvm::StringMap<Value> &outputTensorMap) {
  auto buildMap = [&](StringRef argAttr, StringRef nameAttr,
                      llvm::SmallVector<std::string> &attrsInOrder) {
    llvm::StringMap<std::string> map;
    llvm::SmallVector<std::string> argNames;
    auto funcOp = dyn_cast<func::FuncOp>(op);

    llvm::SmallVector<llvm::StringRef, 2> inputNames;
    auto dictAttr =
        funcOp->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (auto str =
            dictAttr.get(nameAttr).dyn_cast_or_null<mlir::StringAttr>()) {
      str.getValue().split(inputNames, ',', /*MaxSplit=*/-1,
                           /*KeepEmpty=*/false);
    }

    auto argAttrs = funcOp->getAttrOfType<mlir::ArrayAttr>(argAttr);
    if (argAttrs) {
      for (auto attr : argAttrs) {
        auto d = attr.dyn_cast_or_null<mlir::DictionaryAttr>();

        const ArrayRef<Attribute> indexPathAttrs =
            d.get("tf_saved_model.index_path").cast<ArrayAttr>().getValue();
        auto stringAttr =
            indexPathAttrs[0].dyn_cast_or_null<mlir::StringAttr>();
        if (!stringAttr)
          continue;
        argNames.push_back(stringAttr.getValue().str());
      }
    } else {
      for (int i = 0; i < inputNames.size(); i++) {
        argNames.push_back(inputNames[i].str());
      }
    }

    assert(argNames.size() == inputNames.size());
    for (int i = 0; i < inputNames.size(); i++) {
      map[inputNames[i].str()] = argNames[i];
      attrsInOrder.push_back(argNames[i]);
    }
    return map;
  };

  llvm::StringMap<std::string> inNameToAttrMap, outNameToAttrMap;
  llvm::SmallVector<std::string> attrsInOrder;

  inNameToAttrMap = buildMap("arg_attrs", "inputs", attrsInOrder);
  outNameToAttrMap = buildMap("res_attrs", "outputs", attrsInOrder);

  for (int i = 0; i < inNameToAttrMap.size(); i++) {
    inputTensorMap[attrsInOrder[i]] = values[i];
  }

  for (auto v : values) {
    if (auto loc = v.getLoc()->dyn_cast_or_null<NameLoc>()) {
      if (outNameToAttrMap.count(loc.getName())) {
        outputTensorMap[outNameToAttrMap[loc.getName()]] = v;
      }
    }
  }
}

std::vector<int> MemoryPlan::getAllocatedOffsets(const bool overlapOps,
                                                 int &peakMemoryUsed,
                                                 int &peakOpId) {
  std::vector<int> offsets;
  // Copy of valueInfo
  auto vInfo = valueInfo;

  // Overlap buffers
  llvm::DenseMap<Value, std::pair<Value, int>> inOutMap;
  llvm::DenseSet<Operation *> alreadyVisited;
  if (overlapOps) {
    for (auto o : operations) {

      // For async loads, use the same buffer for load and wait
      if (llvm::isa<LoadWeightsWaitOp>(o)) {
        for (int i = 0; i < o->getNumOperands(); i++) {
          auto inVal = o->getOperand(i);
          auto outVal = o->getResult(i);
          vInfo[outVal].firstUsed = vInfo[inVal].firstUsed;
          inOutMap[inVal] = {outVal, 0};
        }
      }

      // We iterate through overlappable ops which have not been visited yet
      if (o->hasTrait<OpTrait::xcore::MemoryOverlappable>() &&
          !alreadyVisited.contains(o)) {
        auto inVal = o->getOperand(0);

        // We have binary and unary ops as overlappable
        // For binary ops, we might have to overlap with the second operand
        // The complicated if condition below is to check for valid one operand
        // or two operand cases
        if ((o->getNumOperands() == 1 && inVal.hasOneUse() &&
             !vInfo[inVal].isConstant) ||
            (o->getNumOperands() == 2 &&
             (inVal.hasOneUse() && !vInfo[inVal].isConstant ||
              o->getOperand(1).hasOneUse() &&
                  !vInfo[o->getOperand(1)].isConstant))) {
          // In case of two operands and first operand is invalid, use the
          // second one
          if (o->getNumOperands() == 2 &&
              (!inVal.hasOneUse() || vInfo[inVal].isConstant)) {
            inVal = o->getOperand(1);
          }

          alreadyVisited.insert(o);
          llvm::SmallVector<Value> inputVals;
          inputVals.push_back(inVal);

          auto outVal = o->getResult(0);

          // Only overlap if the output value size is equal or larger than the
          // input value size We use the allocated space for the output value to
          // store the input value
          if ((utils::getShapedTypeSize(
                   outVal.getType().dyn_cast<ShapedType>()) >=
               utils::getShapedTypeSize(
                   inVal.getType().dyn_cast<ShapedType>()))) {
            auto nextOp = *outVal.getUsers().begin();
            // Identify chain of overlappable Ops
            while (outVal.hasOneUse() && !alreadyVisited.contains(nextOp) &&
                   nextOp->hasTrait<OpTrait::xcore::MemoryOverlappable>() &&
                   (utils::getShapedTypeSize(
                        outVal.getType().dyn_cast<ShapedType>()) >=
                    utils::getShapedTypeSize(
                        inVal.getType().dyn_cast<ShapedType>()))) {
              inVal = outVal;
              inputVals.push_back(inVal);
              alreadyVisited.insert(nextOp);
              outVal = nextOp->getResult(0);
              nextOp = *outVal.getUsers().begin();
            }

            // Set first Used of output Val to the first input Val
            vInfo[outVal].firstUsed = vInfo[inputVals[0]].firstUsed;
            auto unalignedSizeOutVal = utils::getShapedTypeSize(
                outVal.getType().dyn_cast<ShapedType>());
            size_t maxSizeNeeded = 0;
            for (auto inV : inputVals) {
              auto unalignedSizeInV = utils::getShapedTypeSize(
                  inV.getType().dyn_cast<ShapedType>());
              auto unalignedOffset = unalignedSizeOutVal - unalignedSizeInV;
              // Align offset up to double word = 8 bytes
              auto offset = ((unalignedOffset + 7) / 8) * 8;
              maxSizeNeeded = std::max(vInfo[inV].size + offset, maxSizeNeeded);
              inOutMap[inV] = {outVal, offset};
            }
            // The aligned input val size plus aligned offset might be larger
            // than aligned output val size
            vInfo[outVal].size = std::max(vInfo[outVal].size, maxSizeNeeded);
          }
        }
      }
    }
  }

  // Handle input output tensor same allocations
  llvm::DenseSet<Value> inputTensorSet;
  llvm::DenseSet<Value> outputTensorSet;
  llvm::StringMap<Value> inputTensorMap, outputTensorMap;

  if (sameAllocationInputOutputTensorOption.size() > 0) {
    buildInputOutputTensorMaps(inputTensorMap, outputTensorMap);
    for (int i = 0; i < sameAllocationInputOutputTensorOption.size();
         i = i + 2) {
      inputTensorSet.insert(
          inputTensorMap[sameAllocationInputOutputTensorOption[i]]);
      outputTensorSet.insert(
          outputTensorMap[sameAllocationInputOutputTensorOption[i + 1]]);
    }
  }

  // The comparator keeps the buffers ordered by id if their sizes are the
  // same
  auto DecreasingSizesComparator = [&](QueueItem &lhs, QueueItem &rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second < rhs.second;
    }
    return vInfo[lhs.first].id < vInfo[rhs.first].id;
  };
  // The top item is the largest one.
  llvm::PriorityQueue<QueueItem, std::vector<QueueItem>,
                      decltype(DecreasingSizesComparator)>
      queue(DecreasingSizesComparator);

  // Insert values and their sizes into priority queue
  // InOutmap prevents adding in values which are overlapped
  // In a chain of overlapped values, only the last value is allocated and the
  // rest are patched up and add in allocated values list later
  // Don't insert same allocation input and output tensors into queue as they
  // are allocated separately
  for (auto v : values) {
    if (!inOutMap.count(v) && !vInfo[v].isConstant &&
        !outputTensorSet.contains(v) && !inputTensorSet.contains(v)) {
      queue.push({v, vInfo[v].size});
    }
  }

  ValuesOrderedByOffset allocatedValues;

  // If there are same allocation input and output tensors, allocate those first
  if (sameAllocationInputOutputTensorOption.size() > 0) {
    // Allocate first input and output tensor with offsets of zero
    allocatedValues.insert(
        {inputTensorMap[sameAllocationInputOutputTensorOption[0]], 0});
    allocatedValues.insert(
        {outputTensorMap[sameAllocationInputOutputTensorOption[1]], 0});

    for (int i = 2; i < sameAllocationInputOutputTensorOption.size();
         i = i + 2) {
      auto inputTensor =
          inputTensorMap[sameAllocationInputOutputTensorOption[i]];
      int newOffset = getOffset(inputTensor, vInfo[inputTensor].size, vInfo,
                                allocatedValues);
      allocatedValues.insert({inputTensor, newOffset});
      allocatedValues.insert(
          {outputTensorMap[sameAllocationInputOutputTensorOption[i + 1]],
           newOffset});
    }
  } else {
    // Else allocate the largest tensor at offset zero
    auto v = queue.top().first;
    queue.pop();
    allocatedValues.insert({v, 0});
  }

  while (!queue.empty()) {
    auto v = queue.top().first;
    auto size = queue.top().second;
    queue.pop();

    int newOffset = getOffset(v, size, vInfo, allocatedValues);
    allocatedValues.insert({v, newOffset});
  }

  // Patch up overlapped buffers
  for (auto val : inOutMap) {
    auto in = val.first;
    auto out = val.second.first;
    auto offset = val.second.second;

    auto it = std::find_if(allocatedValues.begin(), allocatedValues.end(),
                           [&](const QueueItem &p) { return p.first == out; });

    if (it != allocatedValues.end()) {
      int currentOffset = it->second;
      allocatedValues.insert({in, currentOffset + offset});
    } else {
      assert(false);
    }
  }

  // Insert -1 offset for constant values
  for (auto v : values) {
    if (vInfo[v].isConstant) {
      allocatedValues.insert({v, -1});
    }
  }

  // Sort the allocated offsets by id, i.e., execution order
  auto cmp = [&](QueueItem a, QueueItem b) {
    return vInfo[a.first].id < vInfo[b.first].id;
  };
  std::multiset<QueueItem, decltype(cmp)> allocatedValuesOrderedByID(cmp);
  for (auto i : allocatedValues) {
    allocatedValuesOrderedByID.insert(i);
  }

  // Check if buffers clash
  // for (auto i : allocatedValuesOrderedByID) {
  //   for (auto j : allocatedValuesOrderedByID) {
  //     if (vInfo[i.first].id < vInfo[j.first].id) {
  //       if ((vInfo[i.first].firstUsed > vInfo[j.first].firstUsed &&
  //            vInfo[i.first].firstUsed < vInfo[j.first].lastUsed) ||
  //           (vInfo[j.first].firstUsed > vInfo[i.first].firstUsed &&
  //            vInfo[j.first].firstUsed < vInfo[i.first].lastUsed)) {
  //         auto iBegin = i.second;
  //         auto iEnd = i.second + vInfo[i.first].size;
  //         auto jBegin = j.second;
  //         auto jEnd = j.second + vInfo[j.first].size;
  //         if ((iBegin > jBegin && iBegin < jEnd) ||
  //             (jBegin > iBegin && jBegin < iEnd)) {
  //           printf("\n\nProblem!");
  //           std::cout << "\nValue one " << vInfo[i.first].id
  //                     << ", size = " << vInfo[i.first].size
  //                     << ", offset = " << i.second
  //                     << ", first = " << vInfo[i.first].firstUsed
  //                     << ", last = " << vInfo[i.first].lastUsed;
  //           std::cout << "\nValue two " << vInfo[j.first].id
  //                     << ", size = " << vInfo[j.first].size
  //                     << ", offset = " << j.second
  //                     << ", first = " << vInfo[j.first].firstUsed
  //                     << ", last = " << vInfo[j.first].lastUsed;
  //         }
  //       }
  //     }
  //   }
  // }

  size_t peakUsed = 0;
  size_t peakUsedValueID = 0;
  size_t maxId = 0;
  nonConstantAllocatedValues.clear();
  nonConstantOffsets.clear();
  LLVM_DEBUG(llvm::dbgs() << "\nAllocated offsets : ");
  for (auto i : allocatedValuesOrderedByID) {
    offsets.push_back(i.second);
    if (!vInfo[i.first].isConstant) {
      maxId++;
      nonConstantAllocatedValues.push_back(i.first);
      nonConstantOffsets.push_back(i.second);
      size_t currentSize = vInfo[i.first].size + i.second;
      if (currentSize >= peakUsed) {
        peakUsed = currentSize;
        peakOpId = maxId;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "\nValue " << vInfo[i.first].id << ", size = "
                            << vInfo[i.first].size << ", offset = " << i.second
                            << ", first = " << vInfo[i.first].firstUsed
                            << ", last = " << vInfo[i.first].lastUsed);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n\nPEAK USED : " << peakUsed << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  peakMemoryUsed = peakUsed;

  // printf("\npeakmemory %d, vid %d maxid %d, opid %d\n", peakMemoryUsed,
  //        vInfo[values[peakUsedValueID]].id, maxId, peakOpId);

  return offsets;
}

char MemoryPlan::getOrdinalCharacter(int i) {
  if (i < 10) {
    return '0' + i;
  } else if (i < 36) {
    return 'a' + (i - 10);
  } else if (i < 62) {
    return 'A' + (i - 36);
  }
  return '*';
}

void MemoryPlan::printMemoryPlan() {
  llvm::outs() << "\nMEMORY PLAN ANALYSIS\n"
               << "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯";

  // llvm::outs() << "\nAllocated Offsets\n";
  // for (int i = 0; i < nonConstantAllocatedValues.size(); ++i) {
  //   llvm::outs() << llvm::format(
  //       "\n%c (id=%d): size=%d, offset=%d, first_used=%d last_used=%d",
  //       getOrdinalCharacter(i), i,
  //       valueInfo[nonConstantAllocatedValues[i]].size, nonConstantOffsets[i],
  //       valueInfo[nonConstantAllocatedValues[i]].firstUsed,
  //       valueInfo[nonConstantAllocatedValues[i]].lastUsed);
  // }
  // llvm::outs() << "\n";

  // llvm::outs() << "\nMemory Plan\n";

  constexpr int kLineWidth = 60;
  int max_size = kLineWidth;
  int max_time = 0;
  for (int i = 0; i < nonConstantAllocatedValues.size(); ++i) {
    const int offset = nonConstantOffsets[i];
    const int last_time_used =
        valueInfo[nonConstantAllocatedValues[i]].lastUsed;
    const int size = offset + valueInfo[nonConstantAllocatedValues[i]].size;
    if (size > max_size) {
      max_size = size;
    }
    if (last_time_used > max_time) {
      max_time = last_time_used;
    }
  }

  char line[kLineWidth + 1];
  for (int t = 0; t <= max_time; ++t) {
    for (int c = 0; c < kLineWidth; ++c) {
      line[c] = '.';
    }
    int memory_use = 0;
    int peakSize = 0;
    for (int i = 0; i < nonConstantAllocatedValues.size(); ++i) {
      if ((t < valueInfo[nonConstantAllocatedValues[i]].firstUsed) ||
          (t > valueInfo[nonConstantAllocatedValues[i]].lastUsed)) {
        continue;
      }
      const int offset = nonConstantOffsets[i];
      if (offset == -1) {
        continue;
      }

      const int size = valueInfo[nonConstantAllocatedValues[i]].size;
      if (peakSize < offset + size) {
        peakSize = offset + size;
      }

      memory_use += size;
      const int line_start = (offset * kLineWidth) / max_size;
      const int line_end = ((offset + size) * kLineWidth) / max_size;
      for (int n = line_start; n < line_end; ++n) {
        if (line[n] == '.') {
          line[n] = getOrdinalCharacter(i);
        } else {
          line[n] = '!';
        }
      }
    }
    line[kLineWidth] = 0;

    llvm::outs() << llvm::format(
        "\n%-20s %s%d: %s (%dk), (%dk)",
        operations[t]->getName().stripDialect().str().c_str(),
        t < 10 ? " " : "", t, (const char *)line, (memory_use + 1023) / 1024,
        (peakSize + 1023) / 1024);
  }
  llvm::outs() << "\n";
}

int MemoryPlan::getNextBottomOpId(int opId) {
  Block *block = &op->getRegion(0).front();
  const LivenessBlockInfo *lvb = liveness.getLiveness(block);
  Operation *startOp = lvb->getStartOperation(nonConstantAllocatedValues[opId]);
  Operation *endOp =
      lvb->getEndOperation(nonConstantAllocatedValues[opId], startOp);
  int nextOpId = operationIds[endOp];

  if (nextOpId < opId) {
    nextOpId = -1;
  } else if (nextOpId == opId) {
    nextOpId++;
  }

  if (nextOpId != -1) {
    startOp = lvb->getStartOperation(nonConstantAllocatedValues[nextOpId]);
    endOp = lvb->getEndOperation(nonConstantAllocatedValues[nextOpId], startOp);
    int nextNextOpId = operationIds[endOp];
    if (nextNextOpId != nextOpId) {
      nextOpId = nextNextOpId;
    }
  }

  return nextOpId;
}

} // namespace mlir::xcore
