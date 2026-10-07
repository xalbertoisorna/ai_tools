Example without flash
=====================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

Ensure ``xcore-opt`` is on your ``PATH``. CMake includes ``export.cmake``
to export the model during configuration; no Python export step is needed.

In order to compile and run this example follow these steps::

  # For XS3 (XCORE.AI)
  cmake -G "Unix Makefiles" -B build
  # For VX4 (XCORE-400), use this configure command instead
  cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=XK-EVK-XU416
  xmake -C build
  xrun --xscope bin/app_no_flash.xe

When run, the program should print something similar to::

  No human (9%)
  Human (98%)

The CMake configure step optimises the ``vww_quant.tflite`` model for xcore;
it produces three files::

  src/model.tflite
  src/model.tflite.cpp
  src/model.tflite.h

The first file contains the optimised model,
the second file contains the generated source code, and
the third file contains the header for the source code.

The configure and build steps build the project.

The final step runs the code.
