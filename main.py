from PIL import Image
import cv2

im_file = "screenshot-2025-10-29_13-04-39.png"

im = Image.open(im_file)
im.show()

img = cv2.imread(im_file)
