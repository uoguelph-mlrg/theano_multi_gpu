'''
Trying to reproduce the errors of memcpy_peer

1st step use only random generator and pycuda

on gpu1 run
python toy_p2p_error.py 1 2

on gpu11 run
python toy_p2p_error.py 0 2
'''

import sys
from multiprocessing import Process, Queue

import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import time


def fun(shared_args, private_args, this_queue, that_queue):
    drv.init()
    dev = drv.Device(private_args['ind_gpu'])
    ctx = dev.make_context()
    sock = zmq.Context().socket(zmq.PAIR)

    if private_args['flag_client']:
        sock.connect('tcp://localhost:5000')
    else:
        sock.bind('tcp://*:5000')

    np.random.seed(shared_args['seed'])

    mat_local = gpuarray.GPUArray((shared_args['size'], ), 'float32')
    mat_other = gpuarray.GPUArray((shared_args['size'], ), 'float32')

    h = drv.mem_get_ipc_handle(mat_local.ptr)
    shape = mat_local.shape
    dtype = mat_local.dtype

    sock.send_pyobj((shape, dtype, h))
    shape_other, dtype_other, h_other = sock.recv_pyobj()

    mat_remote = gpuarray.GPUArray(shape_other, dtype_other,
                                   gpudata=drv.IPCMemoryHandle(h_other))

    for ind in range(shared_args['num_round']):

        mat_local.set(
            np.random.rand(shared_args['size']).astype('float32'))

        this_queue.put('')
        that_queue.get()

        if private_args['flag_delay']:
            print private_args['ind_gpu'], 'delayed'
            time.sleep(1)

        drv.memcpy_peer(mat_other.ptr,
                        mat_remote.ptr,
                        mat_remote.dtype.itemsize
                        * mat_remote.size,
                        ctx, ctx)
        # ctx.synchronize()
        # bgn_time = time.time()
        # this_queue.put('')
        # that_queue.get()

        # print bgn

        mat_local += mat_other
        mat_local /= 2.

        if not (mat_other.get() == mat_local.get()).all():
            print 'round %d, gpu %d, not equal!!!' % (ind, private_args['ind_gpu'])


if __name__ == '__main__':

    shared_args = {}
    shared_args['num_round'] = 100
    shared_args['seed'] = 1
    shared_args['size'] = 100 * 1000 * 1000

    p_args = {}
    p_args['ind_gpu'] = int(sys.argv[1])
    p_args['verbose'] = True
    p_args['flag_client'] = False
    p_args['flag_delay'] = True

    q_args = {}
    q_args['ind_gpu'] = int(sys.argv[2])
    q_args['verbose'] = True
    q_args['flag_client'] = True
    q_args['flag_delay'] = False

    queue_p = Queue(1)
    queue_q = Queue(1)

    p = Process(target=fun,
                args=(shared_args, p_args, queue_p, queue_q))
    q = Process(target=fun,
                args=(shared_args, q_args, queue_q, queue_p))
    p.start()
    q.start()
    p.join()
    q.join()
