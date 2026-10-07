find_program(XCORE_OPT_EXECUTABLE NAMES xcore-opt REQUIRED)

set(INPUT_MODEL "${CMAKE_CURRENT_LIST_DIR}/mobilenetv2.tflite")
set(OUTPUT_MODEL "${CMAKE_CURRENT_LIST_DIR}/src/model.tflite")
set(EXPORT_OPTS
    "--xcore-thread-count=5"
    "--xcore-conv-err-threshold=3"
    "--xcore-op-split-tensor-arena"
    "--xcore-op-split-top-op=0,7"
    "--xcore-op-split-bottom-op=6,14"
    "--xcore-op-split-num-splits=8,4"
    "--xcore-write-weights-as-array"
    "--xcore-weights-in-external-memory"
    "--xcore-load-externally-if-larger=1500"
    "--xcore-weights-file=${CMAKE_CURRENT_LIST_DIR}/src/model_weights"
)

execute_process(COMMAND
    "${XCORE_OPT_EXECUTABLE}"
    "${INPUT_MODEL}"
    ${EXPORT_OPTS}
    "-o" "${OUTPUT_MODEL}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)