
#include <xcore/chanend.h>

#include "model.tflite.h"
#include "ioserver_default.hpp"

void run(chanend_t io_channel) {
    model_init(NULL);
    model_ioserver_default(io_channel);
}

extern "C" {
    void inferencer(chanend_t io_channel) {
        run(io_channel);
    }
}
