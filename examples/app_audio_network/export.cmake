find_program(XCORE_OPT_EXECUTABLE NAMES xcore-opt REQUIRED)

set(INPUT_MODEL "${CMAKE_CURRENT_LIST_DIR}/denoise_16x8.tflite")
set(OUTPUT_MODEL "${CMAKE_CURRENT_LIST_DIR}/src/model_audioi16.tflite")
set(EXPORT_OPTS "--xcore-thread-count=5")

execute_process(COMMAND
    "${XCORE_OPT_EXECUTABLE}"
    "${INPUT_MODEL}"
    ${EXPORT_OPTS}
    "-o" "${OUTPUT_MODEL}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)