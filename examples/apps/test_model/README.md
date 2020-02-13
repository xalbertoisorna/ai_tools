# Model Testing Example application

This test application will run inference on a user specified TensorFLow Lite for Microcontrollers
model file (.tflite) using the user specified input.

## xCORE

Building for xCORE

    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/test_model.xe path/to/some.tflite path/to/input path/to/output

## x86

Building for x86

    > make TARGET=x86

Running

    > ./bin/test_model path/to/some.tflite path/to/input path/to/output