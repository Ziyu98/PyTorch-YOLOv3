import numpy as np
from numpy.testing import assert_array_equal
import threading
from time import time


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)

def do_dot(a, b):
    #np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    return np.dot(a, b)  # less efficient because the output is stored in a temporary array?


def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func, args=(a_blocks[i, 0, :, :], b_blocks[0, j, :, :]))
            #out_blocks[i, j, :, :] = th.result()
            #th = threading.Thread(target=dot_func, 
            #                      args=(a_blocks[i, 0, :, :], 
            #                            b_blocks[0, j, :, :], 
            #                            out_blocks[i, j, :, :]))
            th.start()
            #out_blocks[i, j, :, :] = th.result()
            threads.append(th)
    c = 0
    for i in range(nblocks):
        for j in range(mblocks):
            th = threads[c]
            out_blocks[i, j, :, :] = th.join()
            #out_blocks[i, j, :, :] = th.get()
            c += 1
    #for th in threads:
    #    th.join()

    return out

if __name__ == '__main__':

    a = np.random.rand(150, 150).astype('float32')

    start = time()
    a1 = pardot(a, a, 3, 3)
    time_par = time() - start
    print('pardot: {:.2f} seconds taken'.format(time_par))
    #for i in range(30):
    #    a1 = pardot(a, a, 3, 3)
    start = time()
    a2 = np.dot(a, a)
    time_dot = time() - start
    #for i in range(500):
    #    a2 = np.dot(a, a)
    print('np.dot: {:.2f} seconds taken'.format(time_dot))
    #assert_array_equal(a1, a2)
