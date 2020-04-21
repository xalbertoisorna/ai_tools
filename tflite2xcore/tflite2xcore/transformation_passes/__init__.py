# Copyright (c) 2020, XMOS Ltd, All rights reserved

from .transformation_passes import *  # TODO: fix this

from .lut_passes import (
    ReplaceTanhPass,
    ReplaceLogisticPass,
    ReplaceReLUPass,
    ReplaceReLU6Pass,
    LegalizeXCLookupTablePass,
)
from .conv2d_passes import (
    Replace1x1Conv2dPass,
    ReplaceDepthwiseConv2dPass,
    ReplaceDeepConv2dPass,
    ParallelizeDeepConv2dPass,
)

from .fully_connected_passes import (
    ReplaceFullyConnectedPass,
    LegalizeXCFullyConnectedWeightPass,
    LegalizeXCFullyConnectedBiasPass,
)

from .pooling_passes import (
    ReplaceMaxPool2DPass,
    ReplaceMaxPool2D2x2Pass,
    ReplaceAveragePool2DPass,
    ReplaceAveragePool2D2x2Pass,
    ReplaceGlobalAveragePool2DPass,
)
from .padding_passes import (
    FuseConv2dPaddingPass,
    SplitPaddingPass,
    FuseConsecutivePadsPass,
)

from .quantize_dequantize_passes import (
    LegalizeQuantizedInputPass,
    LegalizeQuantizedOutputPass,
    LegalizeFloatInputPass,
    LegalizeFloatOutputPass,
)

from .op_version_passes import LegalizeQuantizeVersionPass

from .cleanup_passes import (
    RemoveXCOREWeightBiasOperatorQuantInfo,
    RemoveUnusedBuffersPass,
    RemoveDanglingTensorsPass,
)

from .renaming_passes import LegalizeOperatorOutputTensorNamePass

from .minification_passes import MinifyQuantInfoPass, MinifyTensorNamesPass
