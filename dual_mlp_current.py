import os
import sys
import time
from multiprocessing import Process, Queue


import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def fun_mlp(shared_args, private_args, this_queue, that_queue):

    learning_rate = shared_args['learning_rate']
    n_epochs = shared_args['n_epochs']
    dataset = shared_args['dataset']
    batch_size = shared_args['batch_size']
    L1_reg = shared_args['L1_reg']
    L2_reg = shared_args['L2_reg']
    n_hidden = shared_args['n_hidden']

    # pycuda and zmq environment

    # if shared_args['flag_p2p']:
    if True:
        drv.init()
        dev = drv.Device(private_args['ind_gpu'])
        ctx = dev.make_context()
        sock = zmq.Context().socket(zmq.PAIR)

        if private_args['flag_client']:
            sock.connect('tcp://localhost:5000')
        else:
            sock.bind('tcp://*:5000')

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])

    import theano
    import theano.tensor as T

    from logistic_sgd import load_data
    from mlp import MLP

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    print private_args
    print dataset

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in=28 * 28,
                     n_hidden=n_hidden, n_out=10)

    cost = (classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr)

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]}
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    # setting pycuda and
    # pass handles, only done once

    # if shared_args['flag_p2p']:
    if True:
        if shared_args['flag_combine_params']:

            # define variables
            params_flat = []
            size_list = []
            shape_list = []
            param_ga_list = []

            for param in classifier.params:
                params_flat.append(param.flatten())
                size_list.append(param.get_value().size)
                shape_list.append(param.get_value().shape)
                param_ga_list.append(
                    theano.misc.pycuda_utils.to_gpuarray(
                        param.container.value))

            # param_total = theano.shared(
            #     0.5 * np.ones(sum(size_list)).astype('float32'))
            # param_total_other = theano.shared(
            #     np.ones(sum(size_list)).astype('float32'))

            # compile functions, concatenate, distribute, average
            # concatenate = theano.function(
            #     [], updates=[(param_total, T.concatenate(params_flat))])
            # concatenate = theano.function(
            #     [], updates=[(param_total, param_total)])

            # updates = []
            # bgn_loc = 0
            # for param, size, shape in zip(classifier.params,
            #                               size_list, shape_list):
            #     updates.append((
            #         param,
            #         param_total[bgn_loc:bgn_loc + size].reshape(shape)))
            #     bgn_loc += size
            # distribute = theano.function([], updates=updates)

            # average = theano.function(
            #     [], updates=[(param_total,
            #                   (param_total + param_total_other) / 2.)])

            # pass handle
            # param_total_ga = theano.misc.pycuda_utils.to_gpuarray(
            #     param_total.container.value)

            # param_total_ga_other = theano.misc.pycuda_utils.to_gpuarray(
            #     param_total_other.container.value)

            param_total_ga = gpuarray.GPUArray((sum(size_list), ),
                                               'float32')
            param_total_ga_other = gpuarray.GPUArray((sum(size_list), ),
                                                     'float32')

            h = drv.mem_get_ipc_handle(param_total_ga.ptr)
            shape = param_total_ga.shape
            dtype = param_total_ga.dtype

            sock.send_pyobj((shape, dtype, h))
            shape_other, dtype_other, h_other = sock.recv_pyobj()

            param_total_ga_remote = \
                gpuarray.GPUArray(shape_other, dtype_other,
                                  gpudata=drv.IPCMemoryHandle(h_other))

            sync_buffer = np.empty_like(param_total_ga, dtype='float32')
            # sync_buffer = np.empty_like(1, dtype='float32')

        else:
            param_ga_list = []
            param_other_list = []
            param_ga_other_list = []
            h_list = []
            shape_list = []
            dtype_list = []
            average_fun_list = []

            for param in classifier.params:
                param_other = theano.shared(param.get_value())
                param_ga = theano.misc.pycuda_utils.to_gpuarray(
                    param.container.value)
                param_ga_other = \
                    theano.misc.pycuda_utils.to_gpuarray(
                        param_other.container.value)
                h = drv.mem_get_ipc_handle(param_ga.ptr)
                average_fun = \
                    theano.function([],
                                    updates=[(param,
                                              (param + param_other) / 2.)])

                param_other_list.append(param_other)
                param_ga_list.append(param_ga)
                param_ga_other_list.append(param_ga_other)
                h_list.append(h)
                shape_list.append(param_ga.shape)
                dtype_list.append(param_ga.dtype)
                average_fun_list.append(average_fun)

            sock.send_pyobj((shape_list, dtype_list, h_list))
            shape_other_list, dtype_other_list, h_other_list = \
                sock.recv_pyobj()

            param_ga_remote_list = []
            for shape_other, dtype_other, h_other in zip(shape_other_list,
                                                         dtype_other_list,
                                                         h_other_list):
                param_ga_remote = \
                    gpuarray.GPUArray(shape_other, dtype_other,
                                      gpudata=drv.IPCMemoryHandle(h_other))

                param_ga_remote_list.append(param_ga_remote)

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    this_queue.put('')
    that_queue.get()
    start_time = time.time()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            if minibatch_index % 2 == private_args['mod']:
            # if True:
                train_model(minibatch_index)

                this_queue.put('')
                that_queue.get()

                # exchaning weights through Queue and calculation through CPU
                if shared_args['flag_p2p'] and \
                        shared_args['flag_combine_params']:

                    # concatenate, send, average, distribute
                    # concatenate()

                    # val_0 = param_total.get_value()
                    # distribute()
                    # concatenate()
                    # val_1 = param_total.get_value()
                    # print np.any(val_0 != val_1)

                    # pycuda concatenate
                    bgn_ptr = 0
                    for param in param_ga_list:
                        this_size = param.dtype.itemsize * param.size
                        drv.memcpy_dtod(param_total_ga.ptr + bgn_ptr,
                                        param.ptr, this_size)
                        
                        # # do dtoh to force syncronization
                        # drv.memcpy_dtoh(sync_buffer,
                        #                 # param_total_ga.ptr + bgn_ptr)
                        #                 param_total_ga.ptr)

                        bgn_ptr += this_size


                    this_queue.put('')
                    that_queue.get()
                    # time.sleep(0.1)

                    # param_total.set_value(param_total.get_value() + 0.1)


                    drv.memcpy_peer(param_total_ga_other.ptr,
                                    param_total_ga_remote.ptr,
                                    param_total_ga_remote.dtype.itemsize
                                    * param_total_ga_remote.size,
                                    ctx, ctx)
                    # do dtoh to force syncronization
                    drv.memcpy_dtoh(sync_buffer,
                                    param_total_ga_other.ptr)

                    # this_queue.put('')
                    # that_queue.get()
                    # time.sleep(0.1)

                    # Change these names and build better debugging "gates"

                    # print 'ph %d' % private_args['ind_gpu'], param_total_ga
                    # print 'po %d' % private_args['ind_gpu'], param_total_ga_other

                    # print 'th %d' % private_args['ind_gpu'], param_total.get_value()
                    # print 'to %d' % private_args['ind_gpu'],
                    # param_total_other.get_value()

                    # average()

                    # print 'ta %d' % private_args['ind_gpu'], param_total.get_value()
                    # print 'pa %d' % private_args['ind_gpu'], param_total_ga

                    # distribute()

                    # pycuda average
                    param_total_ga += param_total_ga_other
                    param_total_ga /= 2.

                    # pycuda distribute
                    bgn_ptr = 0
                    for param in param_ga_list:
                        this_size = param.dtype.itemsize * param.size
                        drv.memcpy_dtod(param.ptr,
                                        param_total_ga.ptr + bgn_ptr,
                                        this_size)
                        bgn_ptr += this_size

                elif shared_args['flag_p2p'] and \
                        (not shared_args['flag_combine_params']):
                    for param_ga_other, param_ga_remote in \
                            zip(param_ga_other_list,
                                param_ga_remote_list):

                        drv.memcpy_peer(param_ga_other.ptr,
                                        param_ga_remote.ptr,
                                        param_ga_remote.dtype.itemsize *
                                        param_ga_remote.size,
                                        ctx, ctx)
                        # time.sleep(0.001)
                    this_queue.put('')
                    that_queue.get()

                    # for debugging exchange weights again and verifying
                    # print ' '
                    # for param, param_other in zip(classifier.params,
                    #                               param_other_list):
                    #     this_W_val = param_other.get_value()
                    #     this_queue.put(param.get_value())
                    #     that_W_val = that_queue.get()

                    #     prev_val = np.copy(that_W_val)

                    #     print private_args['gpu'], np.any(this_W_val != that_W_val)
                    #     if np.any(this_W_val != that_W_val):
                    #         print this_W_val
                    #         print that_W_val
                    #         print prev_val

                    # for param in param_other_list:
                    #     val_for_debug = param.get_value()

                    if shared_args['avg_type'] == 'theano':
                        for average_fun in average_fun_list:
                            average_fun()
                    elif shared_args['avg_type'] == 'cpu':
                        for param, param_other in zip(classifier.params,
                                                      param_other_list):
                            param.set_value((param.get_value() +
                                             param_other.get_value()) / 2.)
                    elif shared_args['avg_type'] == 'pycuda':
                        for param_ga, param_ga_other in \
                                zip(param_ga_list, param_ga_other_list):
                            param_ga += param_ga_other
                            param_ga /= 2.
                    else:
                        raise NotImplementedError(
                            'avg_type can only be theano, cpu or pycuda')

                    # for param in classifier.params:
                    #     val_for_debug = param.get_value()
                else:
                    for param in classifier.params:
                        this_W_val = param.get_value()
                        this_queue.put(this_W_val)
                        that_W_val = that_queue.get()
                        param.set_value((that_W_val + this_W_val) / 2)

                # for debugging exchange weights again and verifying
                # for param in classifier.params:
                #     this_W_val = param.get_value()
                #     this_queue.put(this_W_val)
                #     that_W_val = that_queue.get()
                #     print np.any(this_W_val != that_W_val)

                # test time speed if not actually exchanging weights
                this_queue.put('')
                that_queue.get()

        if private_args['verbose']:
            # for param in classifier.params:
            #     val_for_debug = param.get_value()
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print('%d, epoch %i, minibatch %i/%i, validation error %f %%' %
                  (private_args['ind_gpu'], epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

    end_time = time.time()

    this_queue.put('')
    that_queue.get()

    if private_args['verbose']:
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))
    this_queue.put('')
    that_queue.get()


if __name__ == '__main__':

    shared_args = {}
    shared_args['learning_rate'] = 0.13
    shared_args['n_epochs'] = 10
    shared_args['dataset'] = '/mnt/data/datasets/mnist/mnist.pkl.gz'
    shared_args['batch_size'] = 5000
    shared_args['L1_reg'] = 0.00
    shared_args['L2_reg'] = 0.0001
    shared_args['n_hidden'] = 500
    shared_args['flag_p2p'] = True
    # shared_args['flag_p2p'] = False
    # shared_args['flag_combine_params'] = False
    shared_args['flag_combine_params'] = True
    shared_args['avg_type'] = 'theano'
    # theano
    # pycuda
    # cpu

    p_args = {}
    p_args['ind_gpu'] = 1
    p_args['gpu'] = 'gpu' + str(p_args['ind_gpu'])
    p_args['mod'] = 1
    p_args['verbose'] = True
    p_args['flag_client'] = False

    q_args = {}
    q_args['ind_gpu'] = 0
    q_args['gpu'] = 'gpu' + str(q_args['ind_gpu'])
    q_args['mod'] = 0
    q_args['verbose'] = True
    q_args['flag_client'] = True

    queue_p = Queue(1)
    queue_q = Queue(1)

    p = Process(target=fun_mlp,
                args=(shared_args, p_args, queue_p, queue_q))
    q = Process(target=fun_mlp,
                args=(shared_args, q_args, queue_q, queue_p))
    p.start()
    q.start()
    p.join()
    q.join()
