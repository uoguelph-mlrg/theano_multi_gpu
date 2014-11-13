#Multi-gpu using Theano and PyCUDA

Demonstration of training the same neural network with multipe gpus using Theano and PyCUDA

## Dependencies

Packages

* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [PyCUDA](http://mathema.tician.de/software/pycuda/)
* [zmq](http://zeromq.org/bindings:python)

Files needed to be in the same folder

* [logistic_sgd.py](http://deeplearning.net/tutorial/code/logistic_sgd.py)
* [mlp.py](http://deeplearning.net/tutorial/code/mlp.py)


## Description

* dual_mlp.py : This script trains a multi-layer perceptron with 2 gpus. It uses data parallelism, where 2 minibatches trained separately on 2 gpus are combined to be a larger minibatch. This is by no means the best of way using 2 gpus. The purpose of this code is to show a way of using theano with
multiprocessing and multiple gpus.

* debug/: contains old debug scripts for testing each components, which are poorly documented.

## How to run
In terminal run:
THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python dual_mlp.py arg1 arg2

where arg1 is the index of the 1st gpu and arg2 is the index of the 2nd
gpu. These 2 gpus need to be connected directly by PCI-e, otherwise the
p2p transfer won't work.

For people at University of Guelph, on GPU1~10, run 
THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python dual_mlp.py 1 2

on GPU11, run
THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python dual_mlp.py 0 2

## Acknowledgement
*Frédéric Bastien*, for providing the original page of [Using Multiple GPUs](https://github.com/Theano/Theano/wiki/Using-Multiple-GPUs)
*Lev Givon*, for providing help on inter process communication between 2 gpus with PyCUDA, Lev's original script https://gist.github.com/lebedov/6408165
*Fei Mao* and *Guangyu Sun*, for help on debugging the code