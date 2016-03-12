from theano import tensor as T
from theano.ifelse import ifelse
import theano, time
import numpy as np


if __name__ == "__main__":
    a,b = T.scalars('a', 'b')
    x,y = T.vectors('x', 'y')

    z_lazy = ifelse(T.eq(a, b), # condition
                    T.mean(x),  # then branch
                    T.mean(y))  # else branch

    var_1 = np.array([1, 2])
    var_2 = np.array([3, 4])
    condition_1 = 1
    condition_2 = 1
    iffunction = theano.function([a,b,x,y],[z_lazy])
    result = iffunction(condition_1, condition_2, var_1, var_2)
    print result