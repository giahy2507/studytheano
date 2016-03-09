import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np


'''
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
'''

if __name__ == "__main__":

    np.random.seed(0)

    x = np.random.rand(3,5)

    X = T.dmatrix("x")

    filter_shape = (1 , 1, 2, 2)

    image_shape = (1, 1, -1 , 5)

    layer0 = T.reshape(X, image_shape)

    W = theano.shared(np.asarray(
            np.random.uniform(low=-1, high=1, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

    conv_out = conv.conv2d(input=layer0, filters=W,
                filter_shape=filter_shape, image_shape=image_shape)

    show = theano.function([X], conv_out)

    a = show(x)
    print a.shape


