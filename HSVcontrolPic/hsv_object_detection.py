import cv2
import numpy as np

# callback fucntion for Trackbar returns None
def non(x):
    return None

# assingning Window name for Trackbar
cv2.namedWindow("img")
# creating Trackbar for Lower and Uper value of Hue , Saturation and Value
cv2.createTrackbar('hueL', 'img', 0, 359, non)
cv2.createTrackbar('satL', 'img', 0, 255, non)
cv2.createTrackbar('valL', 'img', 0, 255, non)
cv2.createTrackbar('hueU', 'img', 0, 359, non)
cv2.createTrackbar('satU', 'img', 0, 255, non)
cv2.createTrackbar('valU', 'img', 0, 255, non)

# the main loop
while (True):
    img = cv2.imread('colorballs.jpg')
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    
    # gettind Trackbars' positions and storing it.     
    hueL = cv2.getTrackbarPos('hueL', 'img')
    satL = cv2.getTrackbarPos('satL', 'img')
    valL = cv2.getTrackbarPos('valL', 'img')
    hueU = cv2.getTrackbarPos('hueU', 'img')
    satU = cv2.getTrackbarPos('satU', 'img')
    valU = cv2.getTrackbarPos('valU', 'img')
    
    # converting picture from BGR format to HSV     
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # creating a array of lower and uper HSV value     
    l_b = np.array([hueL, satL, valL])
    u_b = np.array([hueU, satU, valU])
    
    
    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


