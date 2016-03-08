import numpy as np

if __name__ == "__main__":
    a = np.array([[1,2,3],[6,10,1],[3,2,1]])
    print np.argmax(a,axis=1)