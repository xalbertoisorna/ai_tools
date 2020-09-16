# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from tflite2xcore.xcore_schema import (
    QuantizationDetails,
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)
from tflite2xcore.utils import WORD_SIZE

from .transformation_passes import (
    ReplaceWeightBiasOperatorPass,
    LegalizeXCWeightBiasPass,
)


class ReplaceFullyConnectedPass(ReplaceWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc)


class LegalizeXCFullyConnectedPass(LegalizeXCWeightBiasPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_fc

    def _zero_point_bias(self):
        return np.sum(self._weights.as_array(np.int64) * self._input_zero_point, axis=1)

    def mutate_weights(self, op):
        with self.using(op):
            # zero_padding weight tensor
            col_pad = WORD_SIZE - 1 - (self._weights.shape[1] - 1) % WORD_SIZE
            arr = np.pad(self._weights.as_array(), pad_width=[(0, 0), (0, col_pad)])

            self._replace_weights(arr)
