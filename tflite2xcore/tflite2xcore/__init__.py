# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import os
import ctypes
from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = os.path.join(__PARENT_DIR, 'libs/linux/libtflite2xcore.so.1.0.1')
elif sys.platform == "darwin":
    lib_path = os.path.join(__PARENT_DIR, 'libs/macos/libtflite2xcore.1.0.1.dylib')
else:
    lib_path = os.path.join(__PARENT_DIR, 'libs/windows/libtflite2xcore.dll')

libtflite2xcore = ctypes.cdll.LoadLibrary(lib_path)

from . import converter
from . import pass_manager
from . import operator_codes
from . import parallelization
from . import tflite_visualize
from . import utils
from . import xlogging
from . import xcore_model
