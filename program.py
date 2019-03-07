import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

CHARACTER_Q = ord('q')


def find_longest(lines):
    max_length = 0
    longest_line = lines[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance > max_length:
            max_length = distance
            longest_line = line
    return longest_line


def find_line(img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(img, 50, 150, apertureSize=3)
    # blurred = cv2.GaussianBlur(edges_img,(7,7),1)

    min_line_length = 200
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    longest = find_longest(lines)

    # a1, b1, a2, b2 = longest[0]
    # cv2.line(img_orig, (a1, b1), (a2, b2), (0, 0, 255), 5)
    # cv2.imshow("test", img_orig)
    # cv2.waitKey()

    return longest[0]


def find_lines(frame_image):
    img_blue = frame_image[:, :, 0]  # blue channel
    img_green = frame_image[:, :, 1]  # green channel

    # cv2.imshow("b", img_blue)
    # cv2.imshow("g", img_green)

    blue_line = find_line(img_blue)
    green_line = find_line(img_green)
    return blue_line, green_line


cap = cv2.VideoCapture("videos/video-9.avi")
ret, frame = cap.read()

b_line, g_line = find_lines(frame)

i = 0

# mask = cv2.inRange(frame, np.array([160, 160, 160], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
# whiteImage = cv2.bitwise_and(frame, frame, mask=mask)
# cv2.imshow("mask", whiteImage)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", frame_gray)
ret, frame_bin = cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", frame_bin)
frame_numbers = cv2.dilate(frame_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
cv2.imshow("dil", frame_numbers)

# trazenje kontura
_, contours, _ = cv2.findContours(frame_numbers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rectangles = [cv2.boundingRect(contour) for contour in contours]
for rect in rectangles:
    x, y, width, height = rect
    if width > 10:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.drawContours(frame, contours, -1, (255, 0, 0), 1) # sa -1 kazemo da slika svi indekse kontura
cv2.imshow("cont", frame)
print("Ukupan broj kontura: %s" % len(contours))

cv2.waitKey()

# x = [1, 2, 3, 4, 5]
# y = [100, 200, 300, 400, 500]
# plt.plot(x, y)
# plt.show()


"""
while cap.isOpened():
    ret, frame = cap.read()
    # if not ret:
    #     print("Bla bla")
    #     break
    # cv2.imshow('frame', frame)

    i = i + 1
    if cv2.waitKey(10) == CHARACTER_Q or ret is False:
        cap.release()
        cv2.destroyAllWindows()
        break
"""
print(i)

