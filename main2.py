import numpy as np

if __name__ == "__main__":
    filter_shape=(20, 1, 5, 5)
    print filter_shape[1:]
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod((2,2)))


    print fan_in
    print fan_out