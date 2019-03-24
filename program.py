import cv2
import numpy as np
import math
import keras
from neural_network import train_model, get_model
from line import Line
from digit import Digit

CHARACTER_Q = ord('q')
PIXEL_OFFSET = 7


# trazenje najsire konture linije
def find_longest_line(lines):
    max_length = 0
    longest_line = lines[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance > max_length:
            max_length = distance
            longest_line = line
    return longest_line


# trazenje pojedinacne linije
def find_line(img):
    edges_img = cv2.Canny(img, 50, 150, apertureSize=3)

    min_line_length = 200
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    longest = find_longest_line(lines)

    return longest[0]


# trazenje zelene i plave linije
def find_lines(frame_image):
    img_blue = frame_image[:, :, 0]  # blue channel
    img_green = frame_image[:, :, 1]  # green channel

    # cv2.imshow("b", img_blue)
    # cv2.imshow("g", img_green)

    blurred = cv2.GaussianBlur(img_blue, (7, 7), 1)
    # blurred = cv2.dilate(img_blue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
    # cv2.imshow("blurred b", blurred)

    blue_line = find_line(blurred)
    green_line = find_line(img_green)
    return blue_line, green_line


# ispitavanje da li je uoceni broj pre bio detektovan
def already_detect_number(num):
    close_num_objects = []
    for detected_number in detected_numbers:
        distance = np.linalg.norm(np.array(num.bottom_r) - np.array(detected_number.bottom_r))

        if distance < 15:
            close_num_objects.append([detected_number, distance])

    if len(close_num_objects) > 0:
        sorted_close_num = sorted(close_num_objects, key=lambda obj_num: obj_num[1])
        return sorted_close_num[0][0]
    else:
        return None


def out_of_range(num_obj):
    return num_obj.bottom_r[1] > 470 or num_obj.bottom_r[0] > 620


# uocavanje kontura brojeva sa pamcenjem njihovih koordinata
def find_numbers_on_frame(frame):
    # mask = cv2.inRange(frame, np.array([160, 160, 160], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
    # whiteImage = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("mask", whiteImage)
    # procesiranje slike
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", frame_gray)
    ret, frame_bin = cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", frame_bin)
    frame_numbers = cv2.dilate(frame_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
    # cv2.imshow("dil", frame_numbers)

    # trazenje kontura
    _, contours, _ = cv2.findContours(frame_numbers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    for rect in rectangles:
        x, y, width, height = rect
        if height > 10:
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
            digit_o = Digit(x, y, width, height)
            digit_ret = already_detect_number(digit_o)
            if digit_ret is not None:
                digit_ret.update_coordinates(digit_o)
            else:
                detected_numbers.append(digit_o)

    cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)  # sa -1 kazemo da slika svi indekse kontura
    # cv2.imshow("cont", frame)
    return frame_numbers


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation = cv2.INTER_NEAREST)


def scale_to_range(image):  # skalira elemente slike na opseg od 0 do 1
    return image/255


def matrix_to_vector(image):
    return image.flatten()  # svaki element niza/piksela ce se prosledjivati NM


def prepare_for_ann(frame_number, number):
    region = frame_number[number.top_l[1] - PIXEL_OFFSET:number.bottom_l[1] + PIXEL_OFFSET,
             number.bottom_l[0] - PIXEL_OFFSET:number.bottom_r[0] + PIXEL_OFFSET]

    region = resize_region(region)
    region = scale_to_range(region)
    region = matrix_to_vector(region)

    region = region.reshape(1, 28, 28, 1)

    return region


def intersection_over_union(number1, number2):
    (x1_top_1, y1_top_l) = number1.top_l
    (x1_bottom_r, y1_bottom_r) = number1.bottom_r
    (x2_top_1, y2_top_l) = number2.top_l
    (x2_bottom_r, y2_bottom_r) = number2.bottom_r

    x1 = max(x1_top_1, x2_top_1)
    y1 = max(y1_top_l, y2_top_l)
    x2 = min(x1_bottom_r, x2_bottom_r)
    y2 = min(y1_bottom_r, y2_bottom_r)

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    number1_area = (x1_bottom_r - x1_top_1 + 1) * (y1_bottom_r - y1_top_l + 1)
    number2_area = (x2_bottom_r - x2_top_1 + 1) * (y2_bottom_r - y2_top_l + 1)

    iou = intersection_area / float(number1_area + number2_area - intersection_area)

    return iou


# predvidnjanje vrednosti broja pomocu NM
def detect_number_value_of_number_objects(frame_image, frame_numbers):
    # sta sa ivicnim raditi slucajevima kad iskoci izvan prozora....
    # detected_numbers = list(filter(out_of_range, detected_numbers))

    for detect_number in detected_numbers:
        if detect_number.is_detected is False and detect_number.top_l[0] > PIXEL_OFFSET and detect_number.top_l[1] > PIXEL_OFFSET:
            num_for_num = prepare_for_ann(frame_numbers, detect_number)
            v = model.predict(num_for_num)
            v = v.argmax()
            # v = 1
            detect_number.detect_value = v
            detect_number.is_detected = True

        if detect_number.passed_blue_line and not out_of_range(detect_number):
            cv2.rectangle(frame_image, detect_number.top_l, detect_number.bottom_r, (255, 0, 0), 2)
            # cv2.line(frame, blue_line.first_dot(), detect_number.bottom_r, (0, 0, 255), 5)

        if detect_number.passed_green_line and not out_of_range(detect_number):
            cv2.rectangle(frame_image, detect_number.top_l, detect_number.bottom_r, (0, 255, 0), 2)
            # cv2.line(frame, green_line.first_dot(), detect_number.bottom_r, (0, 0, 255), 5)


# ispitivanje da li je presla preko linije
def detect_line_crossing(frame_image, video_name):
    global br_prelaska_plave

    g_first_dot = np.array(green_line.first_dot())
    g_last_dot = np.array(green_line.last_dot())
    b_first_dot = np.array(blue_line.first_dot())
    b_last_dot = np.array(blue_line.last_dot())

    for number in detected_numbers:
        b_dist = np.linalg.norm(np.cross(b_last_dot - b_first_dot, b_first_dot - np.array(number.bottom_r))) / \
                 np.linalg.norm(b_last_dot - b_first_dot)
        g_dist = np.linalg.norm(np.cross(g_last_dot - g_first_dot, g_first_dot - np.array(number.bottom_r))) / \
                 np.linalg.norm(g_last_dot - g_first_dot)

        global result
        global last_added_digit, last_sub_digit
        cv2.putText(frame_image, "iznos=" + str(result), (50, 50), cv2.FONT_ITALIC, 2, (0, 0, 255))
        cv2.putText(frame_image, "+" + str(last_added_digit), (500, 50), cv2.FONT_ITALIC, 2, (255, 0, 0))
        cv2.putText(frame_image, "-" + str(last_sub_digit), (400, 50), cv2.FONT_ITALIC, 2, (0, 255, 0))
        if b_dist < 7 and number.between_line(blue_line) and not number.passed_blue_line:
            number.passed_line = True
            number.passed_blue_line = True
            last_added_digit = number.detect_value
            result += number.detect_value
        if g_dist < 7 and number.between_line(green_line) and not number.passed_green_line:
            number.passed_line = True
            number.passed_green_line = True
            last_sub_digit = number.detect_value
            result -= number.detect_value

    cv2.imshow(video_name, frame_image)


def write_result_to_file():
    with open("txt_result_files/proba.txt", "w") as my_result_file:
        my_result_file.write("RA109/2015 Luka Jovanovic\nfile\tsum")
        for video_number in range(10):
            my_result_file.write("\nvideo-%d.avi\t%d" % (video_number, video_results[video_number]))

# provera da li rade linije? RADI!!!11
# cv2.line(frame, blue_line.first_dot(), blue_line.last_dot(), (255, 0, 255), 5)
# cv2.line(frame, green_line.first_dot(), green_line.last_dot(), (255, 255, 255), 5)
# cv2.imshow("frame", frame)
# cv2.waitKey()

# provera da li radi intersection over union? RADI!!!!!!
# d1 = Digit(3, 3, 2, 2)
# d2 = Digit(1, 1, 3, 2.5)
#
# d3 = Digit(3, 3, 2, 2)
# d4 = Digit(1, 1, 1, 1)
#
# slicnost12 = intersection_over_union(d1, d2)
# print("slicnost12 = ", slicnost12)
# slicnost34 = intersection_over_union(d3, d4)
# print("slicnost34 = ", slicnost34)

# fensi ispis.....
# print(f'x= {x1} a y={y1}')
# print("x= {} a y={}".format(x1, y1))


# --------------------------------------------------------------------------------------------------------------------------

# model = get_model()
# # model = train_model()
# # model = keras.models.load_model("model.h5")
#
# cap = cv2.VideoCapture("videos/video-8.avi")
# ret, frame = cap.read()
#
# b_line, g_line = find_lines(frame)
# blue_line = Line(b_line)
# green_line = Line(g_line)
#
# result = 0
# detected_numbers = []
# last_added_digit = ""
# last_sub_digit = ""
#
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         print("samo iskljucivanje")
#         cv2.waitKey(20)
#         break
#
#     frame_numbers = find_numbers_on_frame(frame)
#     detect_number_value_of_number_objects(frame, frame_numbers)
#     detect_line_crossing(frame, video_name)
#
#     if cv2.waitKey(20) == CHARACTER_Q or ret is False:
#         cap.release()
#         cv2.destroyAllWindows()
#         break
#
# write_result_to_file()

# ------------------------------------------------------------------------------------------------------------------------
model = get_model()
# model = train_model()
# model = keras.models.load_model("model.h5")
result = 0
detected_numbers = []
last_added_digit = ""
last_sub_digit = ""

video_name_pattern = "videos/video-NUMBER.avi"
video_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for video_num in range(10):
    video_name = video_name_pattern.replace("NUMBER", str(video_num))
    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()

    b_line, g_line = find_lines(frame)
    blue_line = Line(b_line)
    green_line = Line(g_line)

    while True:
        video_opened, frame = cap.read()
        if video_opened:
            frame_numbers = find_numbers_on_frame(frame)
            detect_number_value_of_number_objects(frame, frame_numbers)
            detect_line_crossing(frame, video_name)

            if cv2.waitKey(30) == CHARACTER_Q or video_opened is False:
                cap.release()
                cv2.destroyAllWindows()

                video_results[video_num] = result
                result = 0
                detected_numbers = []
                break
        else:
            print("samo iskljucivanje %s" % video_name)
            cap.release()
            cv2.destroyAllWindows()

            video_results[video_num] = result
            result = 0
            detected_numbers = []
            break

write_result_to_file()