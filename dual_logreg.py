import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np


def fun_logreg(shared_args, private_args, this_queue, that_queue):

    learning_rate = shared_args['learning_rate']
    n_epochs = shared_args['n_epochs']
    dataset = shared_args['dataset']
    batch_size = shared_args['batch_size']

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])

    import theano
    import theano.tensor as T

    from logistic_sgd import load_data, LogisticRegression

    print dataset
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    #
    # BUILD ACTUAL MODEL #
    #
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    validate_model = \
        theano.function(inputs=[index],
                        outputs=classifier.errors(y),
                        givens={x: valid_set_x[index * batch_size:
                                               (index + 1) * batch_size],
                                y: valid_set_y[index * batch_size:
                                               (index + 1) * batch_size]})

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # TRAIN MODEL #
    print '... training the model'

    this_queue.put('')
    that_queue.get()
    start_time = time.time()

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            if minibatch_index % 2 == private_args['mod']:
                train_model(minibatch_index)
                # time.sleep(0.05)

                # exchaning weights through Queue and calculation through CPU
                this_W_val = classifier.W.get_value()
                this_queue.put(this_W_val)
                that_W_val = that_queue.get()
                classifier.W.set_value((that_W_val + this_W_val) / 2)
                this_b_val = classifier.b.get_value()
                this_queue.put(this_b_val)
                that_b_val = that_queue.get()
                classifier.b.set_value((that_b_val + this_b_val) / 2)

                # test time speed if not actually exchanging weights
                this_queue.put('')
                that_queue.get()

        if private_args['verbose']:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

    end_time = time.time()

    if private_args['verbose']:
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':

    shared_args = {}
    shared_args['learning_rate'] = 0.13
    shared_args['n_epochs'] = 100
    shared_args['dataset'] = '/mnt/data/datasets/mnist/mnist.pkl.gz'
    shared_args['batch_size'] = 5000

    p_args = {}
    p_args['gpu'] = 'gpu2'
    p_args['mod'] = 1
    p_args['verbose'] = True
    q_args = {}
    q_args['gpu'] = 'gpu1'
    q_args['mod'] = 0
    q_args['verbose'] = False

    queue_p = Queue(1)
    queue_q = Queue(1)

    p = Process(target=fun_logreg,
                args=(shared_args, p_args, queue_p, queue_q))
    q = Process(target=fun_logreg,
                args=(shared_args, q_args, queue_q, queue_p))
    p.start()
    q.start()
    p.join()
    q.join()
