'''
for record the time of copying shared variable arrays using multiprocessing Queue
'''

import time
from multiprocessing import Process, Queue


import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def fcn_send(shared_args, private_args, data_queue, msg_queue):
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    import theano
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    if shared_args['flag_p2p']:
        drv.init()
        dev = drv.Device(private_args['ind_gpu'])
        ctx = dev.make_context()
        sock = zmq.Context().socket(zmq.REQ)
        sock.connect('tcp://localhost:5000')

    W = theano.shared(np.random.rand(shared_args['size']).astype('float32'))


    if shared_args['flag_p2p']:
        W_ga = theano.misc.pycuda_utils.to_gpuarray(W.container.value)
        h = drv.mem_get_ipc_handle(W_ga.ptr)
        sock.send_pyobj((W_ga.shape, W_ga.dtype, h))
        sock.recv_pyobj()

    if shared_args['flag_debug']:
        print W.get_value()
    data_queue.put('')
    msg_queue.get()

    for ind in range(shared_args['num_iter']):
        if shared_args['flag_p2p']:
            msg_queue.get()
        else:
            data_queue.put(W.get_value())
            msg_queue.get()


def fcn_recv(shared_args, private_args, data_queue, msg_queue):
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    import theano
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    if shared_args['flag_p2p']:
        drv.init()
        dev = drv.Device(private_args['ind_gpu'])
        ctx = dev.make_context()

        # # the following lines won't make it faster
        # dev_other = drv.Device(1)
        # ctx_other = dev_other.make_context()

        # ctx_other.pop()
        # ctx.push()
        # ctx.enable_peer_access(ctx_other)

        # ctx.pop()
        # ctx_other.push()
        # ctx_other.enable_peer_access(ctx)

        # ctx_other.pop()
        # ctx.push()
        
        # ctx.enable_peer_access(ctx)

        sock = zmq.Context().socket(zmq.REP)
        sock.bind('tcp://*:5000')

    W = theano.shared(np.zeros(shared_args['size']).astype('float32'))


    if shared_args['flag_p2p']:
        shape, dtype, h = sock.recv_pyobj()
        sock.send_pyobj('')

        W_remote = gpuarray.GPUArray(shape, dtype,
                                     gpudata=drv.IPCMemoryHandle(h))
        W_ga = theano.misc.pycuda_utils.to_gpuarray(W.container.value)

    data_queue.get()
    msg_queue.put('')
    if shared_args['flag_debug']:
        print W.get_value()

    time_bgn = time.time()
    for ind in range(shared_args['num_iter']):
        if shared_args['flag_p2p']:
            drv.memcpy_peer(W_ga.ptr, W_remote.ptr,
                            W_remote.dtype.itemsize * W_remote.size,
                            ctx, ctx)
            msg_queue.put('')
            if shared_args['flag_debug']:
                print W.get_value()
        else:

            W.set_value(data_queue.get())
            msg_queue.put('')

            if shared_args['flag_debug']:
                print W.get_value()

    time_end = time.time()

    time_total = time_end - time_bgn

    time_bgn = time.time()

    for ind in range(shared_args['num_iter']):
        print W.get_value()

    time_end = time.time()

    time_print = time_end - time_bgn

    print 'total time: %.2f' % time_total
    print 'printing time: %.2f' % time_print
    print 'copying time: %.2f' % (time_total - time_print)


if __name__ == '__main__':

    shared_args = {}
    shared_args['flag_p2p'] = True
    shared_args['size'] = 60 * 1000 * 1000
    shared_args['num_iter'] = 10
    shared_args['flag_debug'] = True

    send_args = {}
    send_args['ind_gpu'] = 1
    send_args['gpu'] = 'gpu' + str(send_args['ind_gpu'])
    recv_args = {}
    recv_args['ind_gpu'] = 2
    recv_args['gpu'] = 'gpu' + str(recv_args['ind_gpu'])

    data_queue = Queue(1)
    msg_queue = Queue(1)

    proc_send = Process(target=fcn_send,
                        args=(shared_args, send_args, data_queue, msg_queue))
    proc_recv = Process(target=fcn_recv,
                        args=(shared_args, recv_args, data_queue, msg_queue))

    proc_send.start()
    proc_recv.start()
    proc_send.join()
    proc_recv.join()
