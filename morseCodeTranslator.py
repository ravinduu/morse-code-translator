import numpy as np 
import cv2

image = cv2.imread('images/morse.png',)
print(image.shape)
image = cv2.resize(image,(650,700),interpolation=cv2.INTER_CUBIC)
cv2.imshow('Image',image)	
cv2.waitKey(0)
cv2.destroyAllWindows()

grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Image',grayImage)	
# cv2.waitKey(0)
# cv2.destroyAllWindows()

threshImage = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,45,15)
#_,threshImage = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('Thresh Image',threshImage)	
# cv2.waitKey(0)
# cv2.destroyAllWindows()
kernel = np.ones((5,5),np.uint8)
morphImage = cv2.morphologyEx(threshImage, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Morph Image',morphImage)	
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = np.ones((5,5),np.uint8)
erodeImage = cv2.erode(morphImage,kernel,iterations = 1)
# erodeImage = cv2.dilate(morphImage,kernel,iterations = 1)
cv2.imshow('Erode Image',erodeImage)	
cv2.waitKey(0)
cv2.destroyAllWindows()

contour = cv2.findContours(erodeImage.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
copyImage = np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')

print(contour[0])

for cnt in contour:
    area = cv2.contourArea(cnt)
#     accuracy = 0.1*cv2.arcLength(cnt,True)
#     approx = cv2.approxPolyDP(cnt,accuracy,True)
#     hull   = cv2.convexHull(approx)
#     area = cv2.contourArea(hull)

#     if area<=3000:
#         cv2.drawContours(copyImage,[hull],0,(0,0,255),1)
#         x,y,w,h = cv2.boundingRect(hull)
#         # print(str(w)+','+str(h))
#         cv2.rectangle(copyImage,(x,y),(x+w,y+h),(0,255,0),-1)

# cv2.imshow('Copy Image', copyImage)	
# cv2.waitKey(0)
# cv2.destroyAllWindows()		

# copyImage = copyImage[:,:,1]
# # if self.changeSizeflag == 0:
# #     image = cv2.resize(copyImage,(400,300),interpolation=cv2.INTER_CUBIC)
# # else:	
# # image = cv2.resize(copyImage,(300,200),interpolation=cv2.INTER_CUBIC)
# # self.finalImage = image.copy()

# # kernel = np.ones((3,3),np.uint8)
# # erodeImage = cv2.dilate(morphImage,kernel,iterations = 1)
# # cv2.imshow('Resize Image',image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()