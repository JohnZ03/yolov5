import cv2
import numpy as np
from pynput.mouse import Button, Controller # simulate mouse movement events and clicking event
import wx # get the screen resolution
import time # timer
import keyboard

mouse = Controller()
#keyboard = Controller()
app = wx.App(False)
# screen resolution
(sx, sy) = wx.GetDisplaySize()
(camx, camy) = (500, 500)

# use cvtColor() method to convert rgb value to a range of hsv value
# as there are lots of variation of one particular color
green = np.uint8([[[0, 0, 255]]])
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# our program will only care about green color object and the rest of the
# colors are ignored
lowerLimit_green = hsvGreen[0][0][0] - 10, 100, 100
upperLimit_green = hsvGreen[0][0][0] + 10, 255, 255
#lowerLimit = np.array(lowerLimit)
#upperLimit = np.array(upperLimit)

# In the morning (Table lamp only)
# lowerLimit_green = np.array([20,90,120])
# upperLimit_green = np.array([100,255,255])
lowerLimit_green = np.array([30,50,100])
upperLimit_green = np.array([100,255,255])

# initialize camera object
cam = cv2.VideoCapture(0)

# filtering the mask to eliminate the noise
# morphological operation: opening
# remove all the dots randomly poping 
kernalOpen = np.ones((5,5))
# morphological operation: closing
# close the small holes that are present in the actual object
kernalClose = np.ones((20, 20))

pinchFlag = 0

midLine = camy / 2

gameOver = False

while not gameOver:
    if keyboard.is_pressed('q'):
        gameOver = True

    # read a frame from the camera
    ret, img = cam.read()

    # resize the image frame to a small size for faster processing
    img = cv2.resize(img, (camx, camy))

    # convert the image to hsv format
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create a binary image of same size of original image
    # only those pixels that are in the hsv range will be 
    # displayed in this mask
    # create a mask for green color
    mask = cv2.inRange(imgHSV, lowerLimit_green, upperLimit_green)

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernalOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernalClose)

    # draw contours from mask
    maskFinal = maskClose
    # RETR_EXTERNAL flag to get the ourter most contour of the shape
    # CHAIN_APPROX_NONE stoes every pixel and CHAIN_APPROX_SIMPLE store only endpoints of the line
    # that forms the contour
    conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # draw all contours in an image
    #cv2.drawContours(img, conts, -1, (255, 0, 0), 3)

    # for i in range(len(conts)):
    #     x,y,w,h = cv2.boundingRect(conts[i])
    #     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

    # draw middle line in the capture image
    #cv2.line(img, (0, int(camy/2)), (camx, int(camy/2)), (0, 0, 255), 2)

    if len(conts) == 2:
        #mouse.release(Button.left)

        if pinchFlag == 1:
            pinchFlag = 0
            mouse.release(Button.left)

        # finger open gesture, move mouse without click
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])

        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

        cx1 = int(x1+w1/2)
        cy1 = int(y1+h1/2)
        cx2 = int(x2+w2/2)
        cy2 = int(y2+h2/2)

        cx = int((cx1 + cx2) / 2)
        cy = int((cy1 + cy2) / 2)

        # if cy < midLine - 50:
        #     mouse.scroll(0, 0.4)
        # elif cy > midLine + 20:
        #     mouse.scroll(0, -0.4)

        cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)

        #mouse.release(Button.left)
        mouseLoc = (int(sx-(cx*sx/camx)), int(cy*sy/camy))
        mouse.position = mouseLoc
        while mouse.position != mouseLoc:
            pass

    elif len(conts) == 1:
        # finger close gesture
        x,y,w,h=cv2.boundingRect(conts[0])

        #mouse.press(Button.left)

        if pinchFlag == 0:
            pinchFlag = 1
            mouse.press(Button.left)


        # draw rectangle
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cx = int(x + w/2)
        cy = int(y + h/2)
        cv2.circle(img, (cx, cy), int((w+h)/4), (255, 255, 0), 2)

        #mouse.press(Button.left)
        mouseLoc = (int(sx-(cx*sx/camx)), int(cy*sy/camy))
        mouse.position = mouseLoc
        while mouse.position != mouseLoc:
            pass

    #cv2.imshow("mask", mask)
    #cv2.imshow("maskOpen", maskOpen)
    #cv2.imshow("maskClose", maskClose)
    #cv2.moveWindow("cam", 40,30)
    cv2.imshow("cam", img)
    cv2.waitKey(10)