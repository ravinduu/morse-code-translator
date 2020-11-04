import numpy as np 
import cv2
from scanImage import birdEye

morseCodeData = {'01':'A','1000':'B','1010':'C','100':'D','0':'E','0010':'F','110':'G','0000':'H'
                        ,'00':'I','0111':'J','101':'K','0100':'L','11':'M','10':'N','111':'O','0110':'P'
                        ,'1101':'Q','010':'R','000':'S','1':'T','001':'U','0001':'V','011':'W','1001':'X'
                        ,'1011':'Y','1100':'Z'}

def readImage():

    image = cv2.imread('images/10.jpg',)

    image = birdEye(image)
    image = cv2.resize(image,(650,700),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Image',image)	

    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray Image',grayImage)	


    threshImage = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,45,15)
    #_,threshImage = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('Thresh Image',threshImage)	

    kernel = np.ones((5,5),np.uint8)
    morphImage = cv2.morphologyEx(threshImage, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Morph Image',morphImage)	

    kernel = np.ones((5,5),np.uint8)
    erodeImage = cv2.erode(morphImage,kernel,iterations = 1)
    # erodeImage = cv2.dilate(morphImage,kernel,iterations = 1)
    # cv2.imshow('Erode Image',erodeImage)	

    contour = cv2.findContours(erodeImage.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    copyImage = np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')
    # cv2.drawContours(copyImage,contour,-1,(0,0,255),1)
    # cv2.imshow('Con Image',copyImage)	

    for cnt in contour:
        accuracy = 0.0001*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,accuracy,True)
        hull   = cv2.convexHull(approx)
        area = cv2.contourArea(hull)
        if area<=3000:
            # cv2.drawContours(copyImage,[hull],0,(0,0,255),1)
            x,y,w,h = cv2.boundingRect(hull)
            # print(str(w)+','+str(h))
            cv2.rectangle(copyImage,(x,y),(x+w,y+h),(0,255,0),-1)

    # cv2.imshow('Copy Image', copyImage)	
	
    changeSizeflag = 1
    copyImage = copyImage[:,:,1]
    if changeSizeflag == 0:
        image = cv2.resize(copyImage,(400,300),interpolation=cv2.INTER_CUBIC)
    else:	
        image = cv2.resize(copyImage,(300,200),interpolation=cv2.INTER_CUBIC)
    finalImage = image.copy()

    kernel = np.ones((3,3),np.uint8)
    erodeImage = cv2.dilate(morphImage,kernel,iterations = 1)
    cv2.imshow('Resize Image',image)

    decodeMorse(image)


def decodeMorse(image):
    finalData = []
    startCordList = []
    morseList = []
    startCord = (0,0)
    endCord = (0,0)
    output = ''
    for i in range(image.shape[0]):

        balckDotCount = 0
        whiteDotCount = 0
        morseText = ''
        startCord = (0,0)
        
        for j in range(image.shape[1]):
            if image[i,j]==0:
                if whiteDotCount>=30:
                    morseText+='1'
                elif whiteDotCount<30 and whiteDotCount>=10:	
                    morseText+='0'
                balckDotCount+=1
                whiteDotCount = 0

            elif image[i,j]>0:
                if startCord == (0,0):
                    startCord = (i,j)

                balckDotCount = 0
                whiteDotCount+=1

        if morseText!='':
            startCordList.append((startCord[1],startCord[0]))
            morseList.append(morseText)
            startCord = (0,0)

        elif len(morseList)>0:
            currMax = 0
            data = ''
            loc = 0
            for i,item in enumerate(morseList):
                if len(item)>currMax:
                    currMax = len(item)
                    data = item
                    loc = i

            morseList = []
            # print(data)
            finalData.append(data)

        i+=10
    # print(finalData)

    for code in finalData:
        if len(code) < 5:
            output += morseCodeData[code]

    
    print('Translated Message : %s'% output)


def main():
    readImage()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()