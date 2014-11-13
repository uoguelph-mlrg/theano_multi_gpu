'''
See README.md for a short description.
'''

import os
import sys
import time
from multiprocessing import Process, Queue

import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def fun_mlp(shared_args, private_args, this_queue, that_queue):
    '''
    shared_args 
    contains neural network parameters

    private_args
    contains parameters for process run on each gpu

    this_queue and that_queue are used for synchronization between processes.
    '''

    learning_rate = shared_args['learning_rate']
    n_epochs = shared_args['n_epochs']
    dataset = shared_args['dataset']
    batch_size = shared_args['batch_size']
    L1_reg = shared_args['L1_reg']
    L2_reg = shared_args['L2_reg']
    n_hidden = shared_args['n_hidden']

    ####
    # pycuda and zmq environment
    drv.init()
    dev = drv.Device(private_args['ind_gpu'])
    ctx = dev.make_context()
    sock = zmq.Context().socket(zmq.PAIR)

    if private_args['flag_client']:
        sock.connect('tcp://localhost:5000')
    else:
        sock.bind('tcp://*:5000')
    ####

    ####
    # import theano related
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])

    import theano
    import theano.tensor as T

    from logistic_sgd import load_data
    from mlp import MLP

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    ####


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
    ####
    # setting pycuda and
    # pass handles, only done once
    
    param_ga_list = []
    # a list of pycuda gpuarrays which point to value of theano shared variable on this gpu
    
    param_other_list = []
    # a list of theano shared variables that are used to store values of theano shared variable from the other gpu

    param_ga_other_list = []
    # a list of pycuda gpuarrays which point to theano shared variables in param_other_list

    h_list = []
    # a list of pycuda IPC handles

    shape_list = []
    # a list containing shapes of variables in param_ga_list

    dtype_list = []
    # a list containing dtypes of variables in param_ga_list
    
    average_fun_list = []
    # a list containing theano functions for averaging parameters

    for param in classifier.params:
        param_other = theano.shared(param.get_value())
        param_ga = \
            theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_other = \
            theano.misc.pycuda_utils.to_gpuarray(
                param_other.container.value)
        h = drv.mem_get_ipc_handle(param_ga.ptr)
        average_fun = \
            theano.function([], updates=[(param,
                                          (param + param_other) / 2.)])

        param_other_list.append(param_other)
        param_ga_list.append(param_ga)
        param_ga_other_list.append(param_ga_other)
        h_list.append(h)
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)
        average_fun_list.append(average_fun)

    # pass shape, dtype and handles
    sock.send_pyobj((shape_list, dtype_list, h_list))
    shape_other_list, dtype_other_list, h_other_list = sock.recv_pyobj()

    param_ga_remote_list = []

    # create gpuarray point to the other gpu use the passed information
    for shape_other, dtype_other, h_other in zip(shape_other_list,
                                                 dtype_other_list,
                                                 h_other_list):
        param_ga_remote = \
            gpuarray.GPUArray(shape_other, dtype_other,
                              gpudata=drv.IPCMemoryHandle(h_other))

        param_ga_remote_list.append(param_ga_remote)
    ####


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    this_queue.put('')
    that_queue.get()
    start_time = time.time()

    epoch = 0

    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            if minibatch_index % 2 == private_args['mod']:
                train_model(minibatch_index)
                
                this_queue.put('')
                that_queue.get()

                # exchanging weights
                for param_ga, param_ga_other, param_ga_remote in \
                        zip(param_ga_list, param_ga_other_list,
                            param_ga_remote_list):

                    drv.memcpy_peer(param_ga_other.ptr,
                                    param_ga_remote.ptr,
                                    param_ga_remote.dtype.itemsize *
                                    param_ga_remote.size,
                                    ctx, ctx)                
                
                ctx.synchronize()
                this_queue.put('')
                that_queue.get()
                    
                for average_fun in average_fun_list:
                    average_fun()



        if private_args['verbose']:
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
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



if __name__ == '__main__':

    shared_args = {}
    shared_args['learning_rate'] = 0.13
    shared_args['n_epochs'] = 10
    shared_args['dataset'] = '/mnt/data/datasets/mnist/mnist.pkl.gz'
    shared_args['batch_size'] = 5000
    shared_args['L1_reg'] = 0.00
    shared_args['L2_reg'] = 0.0001
    shared_args['n_hidden'] = 500

    p_args = {}
    p_args['ind_gpu'] = int(sys.argv[1])
    p_args['gpu'] = 'gpu' + str(p_args['ind_gpu'])
    p_args['mod'] = 1
    p_args['verbose'] = False
    p_args['flag_client'] = False

    q_args = {}
    q_args['ind_gpu'] = int(sys.argv[2])
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
