Example applications
----------------------------

The no-flash, audio, profiling, two-tile, and DDR examples export their models
with ``xcore-opt`` during CMake configuration. Their ``CMakeLists.txt`` files
include the local export script::

  # export the model
  include(${CMAKE_CURRENT_LIST_DIR}/export.cmake)

Ensure ``xcore-opt`` is on your ``PATH`` before configuring these examples.
No Python model-export step is required. The flash-based and YOLO examples
retain their existing Python export workflows.

These are 6 example models; in order of complexity

* `app_no_flash <app_no_flash/README.rst>`_  - a single model, no flash memory used. This is the
  fastest but most pressure on internal memory.

* `app_flash_single_model <app_flash_single_model/README.rst>`_ - a single model, with learned parameters in
  flash memory. This removes a lot of pressure on internal memory.

* `app_flash_two_models <.app_flash_two_models/README.rst>`_ - two models, with learned parameters in flash memory.

* `app_flash_two_models_one_arena <app_flash_two_models_one_arena/README.rst>`_ - two models, with learned parameters in
  flash memory. The models share a single tensor arena (scratch memory).

* `app_mobilenetv2 <app_mobilenetv2/README.rst>`_ - exporting a MobileNetV2 model (with flash) via xformer, with example inference
  on host (via interpreter) and on device.

* `app_profiling <app_profiling/README.rst>`_ - demonstrates how to enable and use profiling to speed up execution.
