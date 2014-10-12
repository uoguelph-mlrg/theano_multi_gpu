'''
for record the time of copying shared variable arrays using multiprocessing Queue
'''

import time
import sys
import random

from multiprocessing import Process, Queue


import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def fun_transfer(shared_args, private_args, queue_send, queue_recv):
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    import theano
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    print private_args

    if shared_args['flag_p2p']:
        drv.init()
        dev = drv.Device(private_args['ind_gpu'])
        ctx = dev.make_context()
        sock = zmq.Context().socket(zmq.PAIR)

        if private_args['flag_client']:
            sock.connect('tcp://localhost:5000')
        else:
            sock.bind('tcp://*:5000')

    np.random.seed(random.randint(0, 10000))
    W = theano.shared(np.random.rand(shared_args['size']).astype('float32'))
    W_other = theano.shared(np.zeros(shared_args['size']).astype('float32'))

    average_weights = theano.function([], updates=[(W, (W + W_other) / 2.)])

    if shared_args['flag_debug']:
        print 'In gpu%d, this one \n %s' % \
            (private_args['ind_gpu'], np.array_str(W.get_value()))
        print 'In gpu%d, the other \n %s' % \
            (private_args['ind_gpu'], np.array_str(W_other.get_value()))

    # pass handles, only done once
    if shared_args['flag_p2p']:
        W_ga = theano.misc.pycuda_utils.to_gpuarray(W.container.value)
        W_ga_other = theano.misc.pycuda_utils.to_gpuarray(
            W_other.container.value)

        h = drv.mem_get_ipc_handle(W_ga.ptr)
        sock.send_pyobj((W_ga.shape, W_ga.dtype, h))
        shape_other, dtype_other, h_other = sock.recv_pyobj()

        W_ga_remote = gpuarray.GPUArray(shape_other, dtype_other,
                                        gpudata=drv.IPCMemoryHandle(h_other))

    queue_send.put('')
    queue_recv.get()

    print '======== gpu%d start copying ========' % private_args['ind_gpu']

    time_bgn = time.time()
    for ind in range(shared_args['num_iter']):
        # actual data transfer can be many times
        if shared_args['flag_p2p']:
            drv.memcpy_peer(W_ga_other.ptr, W_ga_remote.ptr,
                            W_ga_remote.dtype.itemsize * W_ga_remote.size,
                            ctx, ctx)

            queue_send.put('')
            queue_recv.get()
            average_weights()

        if shared_args['flag_debug']:
            print 'In gpu%d, the other \n %s' % \
                (private_args['ind_gpu'], np.array_str(W_other.get_value()))

    time_end = time.time()
    time_total = time_end - time_bgn

    time_bgn = time.time()

    for ind in range(shared_args['num_iter']):
        if shared_args['flag_debug']:
            print 'In gpu%d, this one \n %s' % \
                (private_args['ind_gpu'], np.array_str(W.get_value()))

    time_end = time.time()

    time_print = time_end - time_bgn

    print 'total time: %.2f' % time_total
    print 'printing time: %.2f' % time_print
    print 'copying time: %.2f' % (time_total - time_print)


if __name__ == '__main__':
    # sys.argv
    # 1 send gpu, 2 recv gpu, 3 size in MB, 4 number of iterations

    shared_args = {}
    shared_args['flag_p2p'] = True
    shared_args['size'] = int(float(sys.argv[3]) * 1000 * 1000)
    shared_args['num_iter'] = int(sys.argv[4])
    shared_args['flag_debug'] = True

    args_0 = {}
    args_0['ind_gpu'] = int(sys.argv[1])
    args_0['gpu'] = 'gpu' + str(args_0['ind_gpu'])
    args_0['flag_client'] = False

    args_1 = {}
    args_1['ind_gpu'] = int(sys.argv[2])
    args_1['gpu'] = 'gpu' + str(args_1['ind_gpu'])
    args_1['flag_client'] = True

    queue_0 = Queue(1)
    queue_1 = Queue(1)

    proc_0 = Process(target=fun_transfer,
                     args=(shared_args, args_0, queue_0, queue_1))
    proc_1 = Process(target=fun_transfer,
                     args=(shared_args, args_1, queue_1, queue_0))

    proc_0.start()
    proc_1.start()
    proc_0.join()
    proc_1.join()
