cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_no_flash_with_ioserver)

# Sandbox configuration
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)
message("XMOS_SANDBOX_DIR: ${XMOS_SANDBOX_DIR}")

# export model and sources
# execute_process(
#   COMMAND python model_build.py
#   WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
#   COMMAND_ERROR_IS_FATAL ANY
# )

# hw and xscope
set(APP_HW_TARGET XK-EVK-XU316)
set(APP_XSCOPE_SRCS src/config.xscope)

# Include directories
set(APP_INCLUDES src)

# Base flags
set(APP_COMPILER_FLAGS
  -report 
  -O3 
  -mcmodel=large 
  -fxscope 
  -Wno-xcore-fptrgroup 
  -g 
  -Xcompiler-xc
  -mtimer-abi=hwtimers 
  -DUSB_TILE=tile[1]
  -DXUD_CORE_CLOCK=600 
  -lquadflash
)

# dependencies and register
set(APP_DEPENDENT_MODULES "lib_ioserver(1.0.0)")
XMOS_REGISTER_APP()

# Custom libs
include(aitools.cmake)
