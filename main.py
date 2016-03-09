import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from tensorflow.examples.tutorials.mnist import input_data

class LenetConvPoolLayer(object):
    def __init__(self, rng , input, input_shape, filter_shape, pool_shape ,activation = T.tanh):
        self.input = input
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape

        assert input_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(pool_shape))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

         # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=input_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=pool_shape, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class FullConectedLayer(object):
    def __init__(self, input, n_in, n_out, activation = T.tanh):
        self.W = theano.shared(value= np.asarray(np.random.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=theano.config.floatX),
                               name = "W",
                               borrow=True)
        self.b = theano.shared(value= np.asarray(np.random.rand(n_out,) ,dtype=theano.config.floatX),
                               name ="b",
                               borrow=True
        )
        self.input = input
        self.ouput = activation(T.dot(input,self.W) + self.b)
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

class SoftmaxLayer(object):

    def __init__(self, input , n_in, n_out):
        self.W = theano.shared(value= np.asarray(np.random.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=theano.config.floatX),
                               name = "W",
                               borrow=True)
        self.b = theano.shared(value= np.asarray(np.random.rand(n_out,) ,dtype=theano.config.floatX),
                               name ="b",
                               borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.input = input
        # parameters of the model
        self.params = [self.W, self.b]

        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)*y)

    def predict(self):
        return self.y_pred

    def error(self,y):
        y = T.argmax(y,1)
         # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    # mnist.train.images.shape  :   (55000, 784)
    # mnist.train.labels        :   (55000) --> [list label ...]

    # next_images, next_labels = mnist.train.next_batch(100)
    # tuple: images, label      :   (100, 784) , (100, 10)

    # minibatch)
    x = T.dmatrix('x')  # data, presented as rasterized images
    y = T.dmatrix('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28

    layer_1 = FullConectedLayer(input=x, n_in=28*28, n_out=100)

    classifier = SoftmaxLayer(input=layer_1.ouput, n_in=100, n_out=10)

    cost = classifier.negative_log_likelihood(y)

    error = classifier.error(y)

    params = layer_1.params + classifier.params

    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(params, gparams):
        updates.append((param, param - 0.1 * gparam))

    train_model = theano.function(inputs=[x,y], outputs=[cost,error],updates=updates)

    counter = 0
    best_valid_err = 100
    early_stop = 20

    batch_size = 100

    epoch_i = 0

    while counter < early_stop:
        epoch_i +=1
        batch_number = int(mnist.train.labels.shape[0]/batch_size)
        for batch in range(batch_number):
            next_images, next_labels = mnist.train.next_batch(100)
            train_cost, train_error = train_model(next_images, next_labels)
        valid_cost, valid_error = train_model(mnist.validation.images, mnist.validation.labels)
        if best_valid_err > valid_error:
            best_valid_err = valid_error
            print "Epoch ",epoch_i, " Validation cost: ", valid_cost, " Validation error: " , valid_error ," ",counter , " __best__ "
            counter = 0
        else:
            counter +=1
            print "Epoch ",epoch_i, " Validation cost: ", valid_cost, " Validation error: " , valid_error ," ",counter



