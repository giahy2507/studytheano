__author__ = 'HyNguyen'

import theano
import theano.tensor as T
import numpy as np
from LayerClasses import MyConvLayer,FullConectedLayer
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    batch_size=100
    number_featuremaps = 20
    sentence_length = 28
    embed_size = 28
    learning_rate = 0.01

    image_shape = (batch_size,1,sentence_length,embed_size)

    filter_shape_encode = (20,1,5,5)
    filter_shape_decode = (1,20,5,5)

    rng = np.random.RandomState(23455)

    # minibatch)
    X = T.dmatrix('X')  # data, presented as rasterized images

    layer0_encode_input = X.reshape((batch_size, 1, 28, 28))
    layer0_encode = MyConvLayer(rng,layer0_encode_input,image_shape=image_shape,filter_shape=filter_shape_encode,border_mode="valid")

    layer1_encode_input = layer0_encode.output.flatten(2)
    layer1_encode_input_shape = (batch_size,layer0_encode.output_shape[1] * layer0_encode.output_shape[2] * layer0_encode.output_shape[3])
    layer1_encode = FullConectedLayer(layer1_encode_input,layer1_encode_input_shape[1],100)

    layer1_decode_input_shape = (batch_size, 100)
    layer2_decode_input = layer1_encode.output
    layer2_decode = FullConectedLayer(layer2_decode_input,100,layer1_encode_input_shape[1])

    layer3_decode_input = layer2_decode.output.reshape(layer0_encode.output_shape)
    layer3_decode = MyConvLayer(rng,layer3_decode_input,image_shape=layer0_encode.output_shape,filter_shape=filter_shape_decode,border_mode="full")

    cost = T.mean(T.sqrt(T.sum(T.sqr(layer3_decode.output.flatten(2) - X), axis=1)), axis=0) + 0.001*(layer0_encode.L2 + layer1_encode.L2 + layer2_decode.L2 + layer3_decode.L2)

    params = layer0_encode.params + layer1_encode.params + layer2_decode.params + layer3_decode.params

    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate* gparam))

    train_model = theano.function(inputs=[X], outputs=[cost, layer3_decode.output],updates=updates)
    valid_model = theano.function(inputs=[X], outputs=[cost, layer3_decode.output])
    show_function = theano.function(inputs=[X], outputs=cost)
    next_images, next_labels = mnist.train.next_batch(batch_size)

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
        next_images, next_labels = mnist.validation.next_batch(batch_size)
        valid_cost, val_out = valid_model(next_images)
        if best_valid_err > valid_cost:
            best_valid_err = valid_cost
            print "Epoch ",epoch_i," Train cost: ", np.mean(np.array(train_costs)), " Validation cost: ", valid_cost, ",counter ",counter , " __best__ "
            counter = 0
        else:
            counter +=1
            print "Epoch ",epoch_i," Train cost: ", np.mean(np.array(train_costs)), " Validation cost: ", valid_cost, ",counter ", counter









