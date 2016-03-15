import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.dmatrix("X")
W = T.dmatrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.dot(v,W) , sequences=X)
compute_elementwise = theano.function(inputs=[X, W], outputs=[results])

# test values
x = np.array([[1,2,3,4]], dtype=theano.config.floatX)
w = np.array([[1,1,1,2],[1,5,1,3]], dtype=theano.config.floatX)

print "x \n", x
print "W \n", w

aaa = compute_elementwise(x, w.T)
print aaa
