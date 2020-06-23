from im2col import im2col_indices
import numpy as np
import time
a = np.random.rand(1, 3, 416, 416)
b = np.random.rand(32, 3, 3, 3)
a_col = im2col_indices(a, 3, 3, 1, 1)
b_col = b.reshape(32, -1)
size = int(a_col.shape[1] / 2)
prev_time = time.time()
res1 = b_col @ a_col
curr_time = time.time()
print('time = %s' % (curr_time - prev_time))
prev_time = time.time()
idx_ = [2 * i for i in range(size)]
a_col_2 = a_col[:, idx_]
res1 = b_col @ a_col_2
curr_time = time.time()
print('time = %s' % (curr_time - prev_time))

