import cv2
import numpy as np
from enlighten_inference import EnlightenOnnxModel
import imgvision as iv

# img = cv2.imread('/home/xuhang/dataset/mono+color/left/16#_10lux_color.bmp')
# p = cv2.ximgproc.guidedFilter(img, img, 20, 9)
#
# cv2.imwrite('p.bmp', p)
#
# img = cv2.imread('./fusion.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#
# img2 = np.empty_like(img)
# img2[:, :, 0] = img[:, :, 0]
# img2[:, :, 1] = cv2.ximgproc.guidedFilter(img[:, :, 0], img[:, :, 1], 20, 20)
# img2[:, :, 2] = cv2.ximgproc.guidedFilter(img[:, :, 0], img[:, :, 2], 20, 20)
#
# img2 = cv2.cvtColor(img2, cv2.COLOR_YCrCb2BGR)
#
# cv2.imwrite('fusion2.png', img2)
#
# a = cv2.PSNR(img, img2)
# #
# # metric = iv.spectra_metric(img, img2)
# # a = metric.PSNR()
# print('a=', a)
# # metric.Evaluation()
#
# model = EnlightenOnnxModel()
# img3 = model.predict(img2)
# cv2.imwrite('fusion3.png', img3)
img1 = cv2.imread('/home/xuhang/dataset/p10/left/c3.jpg')
img2 = cv2.imread('/home/xuhang/dataset/p10/right/m3.jpg')
img1 = cv2.resize(img1, (900, 1200))
img2 = cv2.resize(img2, (900, 1200))

cv2.imwrite('/home/xuhang/dataset/p10/left/c3_low.jpg', img1)
cv2.imwrite('/home/xuhang/dataset/p10/right/m3_low.jpg', img2)

print('hello')
