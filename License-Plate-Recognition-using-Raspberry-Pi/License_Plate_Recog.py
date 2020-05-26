import cv2
import imutils
import numpy as np

img_path = '5.jpg'

img = cv2.imread(img_path)	
img = cv2.resize(img, (620,480))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 170, 200) #Perform Edge detection


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
cropped = gray[topx:bottomx+1, topy:bottomy+1]
cv2.imshow("Gray Image", cropped)

# Find contour in license plate
ret, binImg = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image', binImg)

contours = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]

countContours = 0
for contour in contours:
    x, y, w, h = contourRect = cv2.boundingRect(contour)
    if 100 < w * h < 300:
        countContours += 1
        cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if 600 < w * h < 1000:
        countContours += 1
        cv2.rectangle(cropped, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if 2000 < w * h < 5000:
        countContours += 1
        cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 0, 255), 2)

print("Contours number found: ", countContours)
cv2.imshow("imgg", cropped)
#Read the number plate
#text = pytesseract.image_to_string(Cropped, config='--psm 11')
#print("Detected Number is:",text)

# cv2.imshow('image',img)
# cv2.imshow('Cropped',cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
