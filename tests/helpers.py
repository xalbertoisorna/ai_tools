# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import re

import numpy as np

def compare_tensor_files(expected_file, expected_quantization, predicted_file,
                         predicted_quantization, abs_tol):
    expected_values = np.fromfile(expected_file, dtype='int8') 
    expected_zero_point = expected_quantization.get('zero_point', 0.0)
    expected_scale = expected_quantization.get('scale', 1.0)

    predicted_values = np.fromfile(predicted_file, dtype='int8')
    predicted_zero_point = predicted_quantization.get('zero_point', 0.0)
    predicted_scale = predicted_quantization.get('scale', 1.0)
    predicted_rshift = 0

    len_ratio = len(predicted_values) / len(expected_values)
    if len_ratio == 2:
        # 2 times as many predicted values as expected, reload them as int16
        predicted_values = np.fromfile(predicted_file, dtype='int16')
        predicted_rshift = 8
    elif len_ratio == 4:
        # 4 times as many predicted values as expected, reload them as int32
        predicted_values = np.fromfile(predicted_file, dtype='int32')
        predicted_rshift = 24

    retval = True # until proven otherwise

    for i, (ev, pv) in enumerate(zip(expected_values, predicted_values)):
        tv = pv >> predicted_rshift
        abs_diff = abs(ev-tv)
        if abs_diff > abs_tol:
            print(f'Difference {abs_diff}>{abs_tol}: index={i}, ' \
                f'expected value={ev}, predicted value={tv} {predicted_rshift}')
            retval = False

    return retval
