
import cv2
import imutils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_path = 'bx2.jpg'

img = cv2.imread(img_path)	
img = cv2.resize(img, (620,480))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

screenCnt = None

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break



if screenCnt is None:
	detected = 0
	print ("No contour detected")
else:
	cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = img[topx:bottomx+1, topy:bottomy+1]
cropped_gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
cv2.imshow('abccc', cropped)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

th2 = cv2.adaptiveThreshold(cropped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY, 39, 2)
cv2.imshow("img", th2)
cropped_edges = auto_canny(th2)
ctrs, _ = cv2.findContours(cropped_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sort_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
img_area = cropped.shape[0]*cropped.shape[1]

for i, ctr in enumerate(sort_ctrs):
    print('ctr', ctr)
    x, y, w, h = cv2.boundingRect(ctr)
    print('x', x, 'y', y, 'w',w, 'h', h)
    roi_area = w*h
    roi_ratio = roi_area/img_area
    if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
        if ((h >1.2 * w) and (3 * w >= h)):
            cv2.rectangle(cropped,(x,y),( x + w, y + h ),(90,0,255),1)
            
cv2.imshow("imgg", cropped)
#Read the number plate
#text = pytesseract.image_to_string(Cropped, config='--psm 11')
#print("Detected Number is:",text)

#cv2.imshow('image',img)
#cv2.imshow('Cropped',cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
