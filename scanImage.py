import cv2
import numpy as np

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def birdEye(image):
    image = cv2.resize(image, (420,590))
    
    orig = image.copy()
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()

    contours = cv2.findContours(edged, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break


    approx = rectify(target)
    pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

    M = cv2.getPerspectiveTransform(approx,pts2)
    birdeyeImage = cv2.warpPerspective(orig,M,(800,800))
    birdeyeImage = cv2.resize(birdeyeImage, (420,590))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)

    cv2.imshow("Original.jpg", orig)
    # cv2.imshow("Original Gray.jpg", gray)
    # cv2.imshow("Original Blurred.jpg", blurred)
    # cv2.imshow("Original Edged.jpg", orig_edged)
    cv2.imshow("Outline.jpg", image)
    cv2.imshow("birdeyeImage.jpg", birdeyeImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return birdeyeImage


# image = cv2.imread('images/temp-3.jpg')
# birdEye(image)