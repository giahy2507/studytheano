__author__ = 'HyNguyen'

import theano
import theano.tensor as T
import numpy as np
from main import LenetConvPoolLayer,MyConvPoolLayer,MyUnPoolDeconvLayer
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    batch_size=100

    rng = np.random.RandomState(23455)

    # minibatch)
    X = T.dmatrix('X')  # data, presented as rasterized images

    layer0_input = X.reshape((batch_size, 1, 28, 28))

    layer0 = MyConvPoolLayer(rng,layer0_input,image_shape=(batch_size, 1, 28, 28),filter_shape=(1, 1, 5, 5),k_pool_size=256)
    # layer0.output.shape = (batch_size, 1, 24, 24)
    mask = layer0.mask_k_maxpooling_4D
    layer1 = MyUnPoolDeconvLayer(rng,input=layer0.output,mask_k_maxpooling_4D=mask,input_shape=(batch_size, 1, 24, 24),filter_shape=(1,1,5,5))

    cost = T.mean(T.sqrt(T.sum(T.sqr(layer1.output.flatten(2) - layer0_input.flatten(2)), axis=1)), axis=0)

    OUT = layer1.output

    params = layer0.params + layer1.params
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - 0.01 * gparam))

    train_model = theano.function(inputs=[X], outputs=[cost, layer1.output],updates=updates)

    show_function = theano.function([X], OUT)

    counter = 0
    best_valid_err = 100
    early_stop = 30

    epoch_i = 0

    while counter < early_stop:
        epoch_i +=1
        batch_number = int(mnist.train.labels.shape[0]/batch_size)
        train_costs = []
        for batch in range(batch_number):
            next_images, next_labels = mnist.train.next_batch(batch_size)
            train_cost, train_out = train_model(next_images)
            train_costs.append(train_cost)
            # print train_cost, train_error
        next_images, next_labels = mnist.validation.next_batch(batch_size)
        valid_cost, val_out = train_model(next_images)
        if best_valid_err > valid_cost:
            best_valid_err = valid_cost
            print "Epoch ",epoch_i," Train cost: ", np.mean(np.array(train_costs)), " Validation cost: ", valid_cost, ",counter ",counter , " __best__ "
            counter = 0
        else:
            counter +=1
            print "Epoch ",epoch_i," Train cost: ", np.mean(np.array(train_costs)), " Validation cost: ", valid_cost, ",counter ", counter









