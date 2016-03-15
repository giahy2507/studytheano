__author__ = 'HyNguyen'
def asb():

    # A = np.array([[1,2,3],[3,2,3],[4,2,3]])
    # B = np.array([[False,False,False],[False,False,False],[True,False,False]])
    # A[B] = 0
    # T.bmatrix
    # print B
    # print A

    """
    # Code example for multi hadamart
    A = np.array([[2.0],[2.0],[1.0]])
    print A.shape
    B = np.array([[0],[1],[0]])
    print B.shape
    X = T.dmatrix("X")
    Y = T.dmatrix("Y")
    Z = X * Y
    mul_function = theano.function([X,Y],Z)
    print mul_function(B, A)
    """

    """
    # Code example for argmax & assign value
    X = T.imatrix("X")

    image_shape = (3,3)

    K = X[2:,1:]

    W = theano.shared(np.random.randint(1,10,(2,3)),name="W")

    max_W = T.argmax(W,1)
    changed_W = T.ones(0)
    changed_W = T.set_subtensor(W[range(0,2),max_W],0)

    max_W2 = T.argmax(changed_W,1)
    changed_W = T.set_subtensor(changed_W[range(0,2),max_W2],0)

    changed_W2 = T.ones(0)

    y_pred = T.dot(K,W)

    func_print = theano.function([X], [K,W,y_pred,max_W,changed_W,max_W2,changed_W2])

    k, w , y, max_w,changed_W, max_W2, changed_W2 = func_print([[1,2,3],[1,2,3],[1,2,3]])

    print k
    print W.get_value()
    print y
    print max_w
    print changed_W
    print max_W2
    print changed_W2
    """



    #
    # A = np.array([[1,3,7],[7,9,2],[1,2,1]])
    # zero = np.zeros_like(A)
    # print "zero \n", zero
    # print "A before \n",A
    # B = np.argmax(A,axis=1)
    # print "B before \n",B
    # A[range(3),B] = 0
    # print "A after \n",A
    # B = np.argmax(A,axis=1)
    # print "B after \n",B
import numpy as np
import numpy.linalg as LA
if __name__ == "__main__":
    a = np.arange(12).reshape(3,4)
    b = LA.norm(a,axis=1)
    print a
    print b

