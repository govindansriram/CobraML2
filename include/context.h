//
// Created by sriram on 4/19/25.
//

#ifndef BRARRAYCONTEXT_H
#define BRARRAYCONTEXT_H


namespace cobraml::core {

    enum MemoryDirection {
        HOST_TO_HOST,
        HOST_TO_DEVICE,
        DEVICE_TO_HOST,
        DEVICE_TO_DEVICE,
    };

    /**
     * Tracks all operations (read & write) happening on a memory buffer (if necessary) through a pipeline.
     * This context allows trackable asynchronous operations (assuming device supports it).
     *
     * Current Devices that utilize async operations:
     * - CUDA Streams
     */
    struct BufferContext {
        virtual ~BufferContext() = default;

        /**
         * pauses the host till all operations associated with this buffer have completed.
         */
        virtual void flush() = 0;

        /**
         * checks if the latest operation is compute related
         */
        virtual bool is_compute() = 0;
    };
}

#endif //BRARRAYCONTEXT_H
