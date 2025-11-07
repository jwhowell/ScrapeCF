import cv2
from cv2.typing import MatLike

# Load image
image = cv2.imread("SampleSchedule1A.png")

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_ = cv2.imwrite("temp/img_gray.png", gray)

# blur image
blur = cv2.GaussianBlur(gray, (15, 15), 0)
_ = cv2.imwrite("temp/img_blur15.png", blur)

# set threshold
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
_ = cv2.imwrite("temp/img_thresh.png", thresh)

# set kernal
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
_ = cv2.imwrite("temp/img_kernal.png", kernal)

# dilate image
dilate = cv2.dilate(thresh, kernal, iterations=3)
_ = cv2.imwrite("temp/img_dilate.png", dilate)

# find contours
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h > 50 and w > 20:  # maybe this will fix bounding boxes
        _ = cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

_ = cv2.imwrite("temp/img_bbox.png", image)


# testing without preprocessing at all.

cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h > 50 and w > 20:  # maybe this will fix bounding boxes
        _ = cv2.rectangle(gray, (x, y), (x + w, y + h), (36, 255, 12), 2)

_ = cv2.imwrite("temp/unprocessed_img_bbox.png", image)


def draw_boxes(cvImage: MatLike, file_name: str):
    # Prep image, copy, convert to gray scale, blur, and threshold
    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        _ = cv2.rectangle(cvImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(len(contours))
    filepath = "temp/" + file_name
    _ = cv2.imwrite(filepath, cvImage)


draw_boxes(image, "unprocessed_img_bbox.png")
