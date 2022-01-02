# coding: utf-8

# Useful links
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects
# https://stackoverflow.com/questions/48414823/opencv-feature-matching-multiple-similar-objects-in-an-image/48420702

from collections import namedtuple
import math
import sys
from traceback import print_exc
from time import sleep

import numpy as np
import pyautogui
from matplotlib import pyplot as plt

import cv

# Suppress scientific notation in NumPy printing
np.set_printoptions(suppress=True)

# Matching parameters for corner detection
MIN_MATCH_COUNT = 8
LOWE_MATCH_RATIO = 0.7

# Logging/display configuration - set to True for more verbose output,
# including visual displays of detected objects
LOG_YAW_PITCH_ROLL = False
LOG_DETECTION_MATCHES = False
DISPLAY_SUB_IMAGES_MASK_OPENING = False
DISPLAY_SUB_IMAGES = False
DISPLAY_DETECTED_OBJECTS = False

# Stops image detection on right hand half of screen in live service. Useful
# for displaying debugging without interefering with image detection
DISCARD_HALF_SCREEN_IN_SERVICE = True


def initialise_detector(detector_type):
    assert detector_type in ('sift', 'orb')

    if detector_type == 'sift':
        return cv.SIFT_create()
    if detector_type == 'orb':
        return cv.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)


detector = initialise_detector('sift')

QUERY_NAMES = ('sburg', 'dburg', 'fburg', 'fries', 'drink', 'done')
QUERY_DISPLAY_NAMES = {
    'sburg': 's🍔',
    'dburg': 'd🍔',
    'fburg': 'f🍔',
    'fries': ' 🍟',
    'drink': ' 🥛',
    'done': ' ✅'
}


def initialise_queries():
    # Initialised in a function so as not to pollute the global scope
    queries = {
        QUERY_DISPLAY_NAMES[query_name]: {
            'name': QUERY_DISPLAY_NAMES[query_name],
            'image': cv.imread(f'imgs/{query_name}.png', 0)
        }
        for query_name in QUERY_NAMES
    }

    for query in queries.values():
        query_keypoints, query_descriptors = detector.detectAndCompute(
            query['image'], None)
        query.update(keypoints=query_keypoints, descriptors=query_descriptors)

    return queries


QUERIES = initialise_queries()

TEST_IMG_ORDERS = {
    '1': {
        'sburg',
        'fries',
    },
    '2': {
        'fburg',
    },
    '3': {
        'fburg',
        'fries',
        'drink',
    },
    '4': {
        'sburg',
        'fries',
    },
    '5': {
        'dburg',
        'fries',
    },
    '6': {
        'fburg',
        'fries',
    },
    '7': {
        'sburg',
        'fries',
    },
    '8': {
        'fburg',
        'drink',
    },
    '9': {
        'dburg',
        'drink',
    },
    '10': {
        'dburg',
    },
}

BF_MATCHER = cv.BFMatcher()


def get_order_and_till_sub_images(input_image):
    # https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv#57300889
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html

    hsv_image = cv.cvtColor(input_image.copy(), cv.COLOR_BGR2HSV)

    white = np.array([0, 0, 255], dtype="uint8")

    def get_bounding_rect_of_largest_contour(hsv_image, hsv_lower, hsv_upper):
        mask = cv.inRange(hsv_image, hsv_lower, hsv_upper)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        if DISPLAY_SUB_IMAGES_MASK_OPENING:
            cv.imshow('hsv_image', hsv_image)
            cv.imshow('mask', mask)
            cv.imshow('opening', opening)
            cv.waitKey()

        return cv.boundingRect(biggest_contour)

    order_left, order_top, order_width, order_height = get_bounding_rect_of_largest_contour(
        hsv_image, white, white)

    order_image = input_image[order_top:order_top + order_height,
                              order_left:order_left + order_width]

    order_bottom = order_top + order_height

    till_white_lower = np.array([0, 0, 170], dtype="uint8")
    till_white_upper = np.array([0, 0, 240], dtype="uint8")

    # Assume that the till is always under the bottom of the order
    till_left, till_top, till_width, till_height = get_bounding_rect_of_largest_contour(
        hsv_image[order_bottom:, :], till_white_lower, till_white_upper)

    till_image = input_image[order_bottom + till_top:order_bottom + till_top +
                             till_height, till_left:till_left + till_width]

    if DISPLAY_SUB_IMAGES:
        cv.imshow('order', order_image)
        cv.imshow('till', till_image)
        cv.waitKey()

    return order_image, till_image


def segment_order_sub_image_and_detect_objects(order_image):
    _, order_width, _ = order_image.shape

    hsv_image = cv.cvtColor(order_image.copy(), cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv_image, (0, 0, 0), (255, 255, 254))

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    if DISPLAY_SUB_IMAGES_MASK_OPENING:
        cv.imshow('opening', opening)
        cv.waitKey()

    contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)

    detected_objects = {}

    prev_contour_area = None
    contours_and_areas = ((contour, cv.contourArea(contour))
                          for contour in contours)

    # Sort contours by area, largest first
    for contour, contour_area in sorted(contours_and_areas,
                                        key=lambda c_ca: c_ca[1],
                                        reverse=True):
        left, top, width, height = cv.boundingRect(contour)

        # If contour's bounding box is at far left or right of order sub-image,
        # it's likely to be above/below the speech balloon, so skip
        if left == 0 or left + width == order_width:
            continue

        # If this contour is significantly smaller than the last, it's likely
        # we're passed from images to individual letters so should stop looping
        if prev_contour_area and contour_area < prev_contour_area / 2:
            break

        prev_contour_area = contour_area

        order_item_image = order_image[top:top + height, left:left + width]

        if DISPLAY_SUB_IMAGES:
            cv.imshow('order_item', order_item_image)
            cv.waitKey()

        detected_objects_in_order_item = detect_objects_in_image(
            order_item_image, discard_extreme_angles=True)

        if not detected_objects_in_order_item:
            continue

        # In some cases single/double burgers can both match a single image so
        # we take the best match
        best_match_key = max(
            detected_objects_in_order_item.keys(),
            key=lambda k: detected_objects_in_order_item[k].num_good_matches)

        assert best_match_key not in detected_objects, "Detected same object twice in order"
        detected_objects[best_match_key] = detected_objects_in_order_item[
            best_match_key]

    return detected_objects


def calculate_yaw_pitch_roll_from_homography(homography_matrix):
    """Converts a homography rotation matrix to yaw/pitch/roll angles from
    camera. Useful to discard detections which are angled too far from camera.
    """
    # Useful reference docs
    # http://planning.cs.uiuc.edu/node103.html
    # https://coderedirect.com/questions/83394/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points
    # (Hilmi's answer, while not exactly what was needed, was useful)

    R = homography_matrix

    yaw = math.atan2(R[(1, 0)], R[(0, 0)])
    pitch = math.atan2(-R[(2, 0)], math.hypot(R[(2, 1)], R[(2, 2)]))
    roll = math.atan2(R[(2, 1)], R[(2, 2)])

    if LOG_YAW_PITCH_ROLL:
        print(f"yaw {yaw:f} pitch {pitch:f} roll {roll:f}")

    return yaw, pitch, roll


def display_detected_object(query, input_image, good_matches,
                            cluster_input_keypoints, M, mask):
    """Opens a new matplotlib window to display a detected object and the
    keypoint matches in an original image. Useful for debugging.
    """
    # Flattens array and coerces to list
    matches_mask = mask.ravel().tolist()

    h, w = query['image'].shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv.perspectiveTransform(pts, M)
    input_img_poly = cv.polylines(input_image, [np.int32(dst)], True, 255, 3,
                                  cv.LINE_AA)
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matches_mask,  # draw only inliers
        flags=2)

    match_display_img = cv.drawMatches(query['image'], query['keypoints'],
                                       input_img_poly, cluster_input_keypoints,
                                       good_matches, None, **draw_params)
    plt.imshow(match_display_img, 'gray'), plt.show()


DetectedObject = namedtuple(
    'DetectedObject', 'query, centre_pt, homography_matrix, num_good_matches')


def locate_detected_object(query, input_image, input_keypoints, good_matches,
                           discard_extreme_angles):
    # Useful docs on using and understanding homography
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    # https://www.pythonpool.com/cv2-findhomography/

    src_pts = np.float32([
        query['keypoints'][m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([input_keypoints[m.trainIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    if homography_matrix is None:
        return None

    if discard_extreme_angles:
        yaw, pitch, roll = calculate_yaw_pitch_roll_from_homography(
            homography_matrix)

        # Discard any homography where the image appears to have been reversed
        # or the angle from the camera is too steep.
        # The boundary condition should be `abs(angle) > pi/2` but for some
        # reason the drink was sometimes reporting a "reversed" yaw of 1.9
        # despite appearing correct in displayed homographies, so
        # experimentally set to 2
        if any((abs(angle) > 2 for angle in (yaw, pitch, roll))):
            if LOG_YAW_PITCH_ROLL:
                print("Dicard detected object due to yaw/pitch/roll angle")

                if DISPLAY_DETECTED_OBJECTS:
                    display_detected_object(query, input_image, good_matches,
                                            input_keypoints, homography_matrix,
                                            mask)
            return None

    if DISPLAY_DETECTED_OBJECTS:
        display_detected_object(query, input_image, good_matches,
                                input_keypoints, homography_matrix, mask)

    centre_pt, = np.average(dst_pts, axis=0)

    return DetectedObject(query, centre_pt, homography_matrix,
                          len(good_matches))


def detect_objects_in_image(input_image, discard_extreme_angles):
    input_keypoints, input_descriptors = detector.detectAndCompute(
        input_image, None)

    detected_objects = {}

    for query_name, query in QUERIES.items():
        matches = BF_MATCHER.knnMatch(query['descriptors'], input_descriptors,
                                      2)

        # store all the good_matches matches as per Lowe's ratio test.
        good_matches = [
            m for (m, n) in matches
            if m.distance < LOWE_MATCH_RATIO * n.distance
        ]

        good_matches_count = len(good_matches)
        if good_matches_count < MIN_MATCH_COUNT:
            if LOG_DETECTION_MATCHES:
                print(f"{query_name} - No Match: "
                      f"{len(good_matches)}/{MIN_MATCH_COUNT}")
            continue

        detected_object = locate_detected_object(query, input_image,
                                                 input_keypoints, good_matches,
                                                 discard_extreme_angles)

        if detected_object is not None:
            detected_objects[query_name] = detected_object
            if LOG_DETECTION_MATCHES:
                print(f"{query_name} - Match at {detected_object.centre_pt}: "
                      f"{len(good_matches)}/{MIN_MATCH_COUNT}")
        else:
            if LOG_DETECTION_MATCHES:
                print(f"{query_name} - Match, but no/extreme homography: "
                      f"{len(good_matches)}/{MIN_MATCH_COUNT}")

    return detected_objects


def run_test_detect_objects(args):
    for img_name, expected_order in TEST_IMG_ORDERS.items():
        # Allow selective testing of individual cases
        if args and img_name not in args:
            continue

        expected_order = {
            QUERY_DISPLAY_NAMES[query_name]
            for query_name in expected_order
        }

        input_image = cv.imread(f'imgs/test/{img_name}.png', 0)

        input_image = cv.cvtColor(input_image, cv.COLOR_RGB2BGR)
        order_image, till_image = get_order_and_till_sub_images(input_image)

        order_detected_objects = segment_order_sub_image_and_detect_objects(
            order_image)
        actual_order = order_detected_objects.keys()

        assert expected_order == actual_order, \
            f"""test img {img_name}:
            expected {expected_order}
            actual   {actual_order}"""

        till_detected_objects = detect_objects_in_image(
            till_image, discard_extreme_angles=False)
        actual_till = till_detected_objects.keys()

        expected_till = QUERY_DISPLAY_NAMES.values()

        assert set(expected_till) == set(actual_till), \
            f"""test img {img_name}:
            expected {expected_till}
            actual   {actual_till}"""


def run_test_match_written_screenshot(args):
    input_image = cv.imread('imgs/screen.png', 0)
    detected_objects = detect_objects_in_image(input_image)

    print(f"Matched {tuple(detected_objects.keys())}")


def run_bot_service(args):
    while True:
        sleep(1)

        input_image = pyautogui.screenshot()
        input_image = cv.cvtColor(np.array(input_image), cv.COLOR_RGB2BGR)
        cv.imwrite('imgs/screen.png', input_image)

        if DISCARD_HALF_SCREEN_IN_SERVICE:
            _, input_image_width, _ = input_image.shape
            input_image = input_image[:, :int(input_image_width / 2)]

        try:
            order_image, till_image = get_order_and_till_sub_images(input_image)

            till_detected_objects = detect_objects_in_image(
                till_image, discard_extreme_angles=False)
            print(f"Till: {till_detected_objects.keys()}")

            order_detected_objects = segment_order_sub_image_and_detect_objects(
                order_image)
            print(f"Order: {order_detected_objects.keys()}")

            print()
        except cv.error as e:
            print(f"Skipping Frame: Caught cv.error {e}")
            continue
        except Exception as e:
            print(f"Skipping Frame: Caught exception {e}")
            print_exc()
            continue


if __name__ == '__main__':
    COMMANDS = {
        'test': run_test_detect_objects,
        'test_screen': run_test_match_written_screenshot,
        'bot': run_bot_service
    }

    COMMANDS[sys.argv[1]](sys.argv[2:])
