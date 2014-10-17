'''
test concatenation of different theano shared variable using:
1. theano function
2. gpuarray operation
'''

import theano
import theano.tensor as T
import numpy as np


aaa_send = theano.shared(np.ones((5, 5, 5)).astype('float32'))
bbb_send = theano.shared(np.zeros((727)).astype('float32'))
ccc_send = theano.shared(np.random.rand(1000000).astype('float32'))
ddd_send = theano.shared(np.zeros((5, 5, 5, 64)).astype('float32'))
params_send = [aaa_send, bbb_send, ccc_send, ddd_send, ]

aaa_recv = theano.shared(np.zeros((5, 5, 5)).astype('float32'))
bbb_recv = theano.shared(np.ones((727)).astype('float32'))
ccc_recv = theano.shared(np.random.rand(1000000).astype('float32'))
ddd_recv = theano.shared(np.ones((5, 5, 5, 64)).astype('float32'))
params_recv = [aaa_recv, bbb_recv, ccc_recv, ddd_recv, ]





params_send_flat = []
size_list = []
shape_list = []

for param in params_send:
    params_send_flat.append(param.flatten())
    size_list.append(param.get_value().size)
    shape_list.append(param.get_value().shape)

total_send = theano.shared(np.zeros(sum(size_list)).astype('float32'))


concatenate = \
    theano.function([], updates=[(total_send,
                                  T.concatenate(params_send_flat))])


print total_send.get_value()

concatenate()

print total_send.get_value()


updates = []

bgn_loc = 0
for param, size, shape in zip(params_recv, size_list, shape_list):

    updates.append((param, total_send[bgn_loc:bgn_loc + size].reshape(shape)))
    bgn_loc += size



distribute = theano.function([], updates=updates)

print ''
for param in params_send:
    print param.get_value().flatten()[:10]

print ''
for param in params_recv:
    print param.get_value().flatten()[:10]

distribute()

print ''
for param in params_recv:
    print param.get_value().flatten()[:10]

for param_send, param_recv in zip(params_send, params_recv):
    print np.any(param_send.get_value() != param_recv.get_value())
