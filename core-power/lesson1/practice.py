import cv2

img_bgr = cv2.imread("gugong.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(img_rgb)