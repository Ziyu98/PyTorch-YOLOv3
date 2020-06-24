import numpy as np
import time
from PIL import Image, ImageDraw
"""a = np.random.rand(1, 3, 416, 416)
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
"""
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)).astype('float32')
    p_t = time.time()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    c_t = time.time()
    print('t1:', (c_t - p_t))
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

#X = np.arange(18).reshape(1,2,3,3)
#X_col = im2col(X, 3, 3, stride=1, pad=1)
X = np.arange(3*416*416).reshape(1,3,416,416)
RoIs = [[333, 95, 337, 95, 337, 98, 338, 98, 338, 102, 342, 102, 342, 103, 338, 103, 338, 107, 337, 107, 337, 109, 335, 109, 335, 114, 333, 114, 333, 110, 313, 110, 313, 105, 314, 105, 314, 95, 323, 95, 323, 93, 324, 93, 324, 95, 330, 95, 330, 93, 333, 93], [128, 251, 134, 251, 134, 221, 142, 221, 142, 258, 134, 258, 134, 252, 129, 252, 129, 254, 128, 254, 128, 258, 127, 258, 127, 259, 125, 259, 125, 295, 115, 295, 115, 268, 103, 268, 103, 227, 114, 227, 114, 260, 123, 260, 123, 258, 117, 258, 117, 218, 128, 218], [141, 295, 130, 295, 130, 263, 141, 263]]
for i in range(5):
    X_col = im2col(X, 3, 3, stride=1, pad=1)
    p_t = time.time()
    if RoIs is not None:
        res = np.zeros((416, 416)).astype('float32')
        img = Image.new('L', (416, 416), 0)
        for poly in RoIs:
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        img = np.array(img)
        c_t = time.time()
        print('time for idx 1: ', (c_t - p_t))
        p_t = time.time()
        idx = (np.nonzero(img))
    c_t = time.time()
    print('time for idx 2: ', (c_t - p_t))
    p_t = time.time()
    temp = img[idx]
    res[idx] = temp
    c_t = time.time()
    print('time for idx 3: ', (c_t - p_t))
"""a = np.arange((1* 3* 3*3*416*416)).reshape(1, 3, 3, 3, 416, 416).astype('float32')
p_t = time.time()
a = a.transpose(0, 4, 5, 1, 2, 3).reshape(173056, -1)
c_t = time.time()
print('t1:', (c_t - p_t))
a = np.arange((1* 3* 3*3*70*70)).reshape(1, 3, 3, 3, 70, 70).astype('float32')
p_t = time.time()
a = a.transpose(0, 4, 5, 1, 2, 3).reshape(4900, -1)
c_t = time.time()
print('t2:', (c_t - p_t))"""
