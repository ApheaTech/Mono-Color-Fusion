import numpy as np
import cv2

left_shift = cv2.imread('./left_shift.png')
mono = cv2.imread('/home/xuhang/dataset/mono+color/right/16#_10lux_mono.bmp')

# two-scale image decomposition
# base layer
b1 = cv2.blur(left_shift, (31, 31))
b2 = cv2.blur(mono, (31, 31))
cv2.imwrite('./fusion/b1.png', b1)
cv2.imwrite('./fusion/b2.png', b2)
# detail layer
d1 = left_shift - b1
d2 = mono - b2
cv2.imwrite('./fusion/d1.png', d1)
cv2.imwrite('./fusion/d2.png', d2)

# weight map construction
# laplacian
h1 = cv2.Laplacian(left_shift, 3)
h2 = cv2.Laplacian(mono, 3)
cv2.imwrite('./fusion/h1.png', h1)
cv2.imwrite('./fusion/h2.png', h2)
# saliency map
h1_abs = np.absolute(h1)
h2_abs = np.absolute(h2)
s1 = cv2.GaussianBlur(h1_abs, (11, 11), 5)
s2 = cv2.GaussianBlur(h2_abs, (11, 11), 5)
cv2.imwrite('./fusion/s1.png', s1)
cv2.imwrite('./fusion/s2.png', s2)

p1 = np.zeros((s1.shape[0], s1.shape[1]), dtype='uint8')
p2 = np.zeros((s1.shape[0], s1.shape[1]), dtype='uint8')
for i in range(s1.shape[0]):
    for j in range(s1.shape[1]):
        maxS = max(s1[i, j, 0], s1[i, j, 1], s1[i, j, 2], s2[i, j, 0], s2[i, j, 1], s2[i, j, 2])
        if s1[i, j, 0] == maxS or s1[i, j, 1] == maxS or s1[i, j, 2] == maxS:
            p1[i, j] = 1
        else:
            p1[i, j] = 0
        if s2[i, j, 0] == maxS or s2[i, j, 1] == maxS or s2[i, j, 2] == maxS:
            p2[i, j] = 1
        else:
            p2[i, j] = 0
# p1 = cv2.cvtColor(p1, cv2.COLOR_GRAY2BGR)
# p2 = cv2.cvtColor(p2, cv2.COLOR_GRAY2BGR)
# cv2.imwrite('./fusion/p1.png', p1)
# cv2.imwrite('./fusion/p2.png', p2)

wb1 = cv2.ximgproc.guidedFilter(left_shift, p1, 9, 0.03)
wb2 = cv2.ximgproc.guidedFilter(mono, p2, 9, 0.03)
wd1 = cv2.ximgproc.guidedFilter(left_shift, p1, 5, 0.03)
wd2 = cv2.ximgproc.guidedFilter(mono, p2, 5, 0.03)
cv2.imwrite('./fusion/wb1.png', wb1)
cv2.imwrite('./fusion/wb2.png', wb2)
cv2.imwrite('./fusion/wd1.png', wd1)
cv2.imwrite('./fusion/wd2.png', wd2)

print('hello')
