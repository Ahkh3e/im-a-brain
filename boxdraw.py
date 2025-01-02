import cv2

# Load the image
img = cv2.imread('zoolander.jpg')

# Define the bounding boxes (x, y, w, h)
box1 = (50, 100, 250, 350)  # x, y, width, height
box2 = (300, 150, 550, 450)

# Draw the bounding boxes on the image
cv2.rectangle(img, box1[0:2], box1[2:4], (0, 255, 0), 2)
cv2.rectangle(img, box2[0:2], box2[2:4], (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()