# coding: utf-8

# Useful links
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects
# https://stackoverflow.com/questions/48414823/opencv-feature-matching-multiple-similar-objects-in-an-image/48420702

import math
import sys
from time import sleep

import numpy as np
import pyautogui
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv

# Suppress scientific notation in NumPy printing
np.set_printoptions(suppress=True)

# Matching parameters for corner detection
MIN_MATCH_COUNT = 10
LOWE_MATCH_RATIO = 0.7

# Logging/display configuration - set to True for more verbose output,
# including visual displays of detected objects
LOG_CLUSTER_MATCHES = False
LOG_YAW_PITCH_ROLL = False
DISPLAY_DETECTED_OBJECTS = False


def initialise_detector(detector_type):
    assert detector_type in ('sift', 'orb')

    if detector_type == 'sift':
        return cv.SIFT_create()
    if detector_type == 'orb':
        return cv.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)


detector = initialise_detector('sift')

QUERY_NAMES = ('sburg', 'dburg', 'fburg', 'fries', 'drink', 'done')


def initialise_queries():
    # Initialised in a function so as not to pollute the global scope
    queries = {
        query_name: {
            'name': query_name,
            'image': cv.imread(f'imgs/{query_name}.png', 0)
        }
        for query_name in QUERY_NAMES
    }

    for query_name, query in queries.items():
        query_keypoints, query_descriptors = detector.detectAndCompute(
            query['image'], None)
        query.update(keypoints=query_keypoints, descriptors=query_descriptors)

    return queries


QUERIES = initialise_queries()

TEST_IMG_MATCHES = {
    '1': {
        'sburg': 2,
        'dburg': 1,
        'fburg': 1,
        'fries': 2,
        'drink': 1,
        'done': 1
    },
    '2': {
        'sburg': 1,
        'dburg': 1,
        'fburg': 2,
        'fries': 1,
        'drink': 1,
        'done': 1
    },
    '3': {
        'sburg': 1,
        'dburg': 1,
        'fburg': 2,
        'fries': 2,
        'drink': 2,
        'done': 1
    },
    '4': {
        'sburg': 2,
        'dburg': 1,
        'fburg': 1,
        'fries': 2,
        'drink': 1,
        'done': 1
    },
    '5': {
        'sburg': 1,
        'dburg': 2,
        'fburg': 1,
        'fries': 2,
        'drink': 1,
        'done': 1
    },
}

BF_MATCHER = cv.BFMatcher()


def create_meanshift(keypoints):
    """Meanshifts split keypoints into clusters so that we can try to detect
    multiple objects in a single image.
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html

    x = np.array([keypoints[0].pt])

    for i in range(len(keypoints)):
        x = np.append(x, [keypoints[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    return ms


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


def locate_detected_object_centre(query, input_image, input_keypoints,
                                  input_descriptors, good_matches,
                                  descriptor_indexes):
    # Useful docs on using and understanding homography
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    # https://www.pythonpool.com/cv2-findhomography/

    cluster_input_keypoints = [
        input_keypoints[index] for index in descriptor_indexes
    ]

    src_pts = np.float32([
        query['keypoints'][m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([
        cluster_input_keypoints[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    homography_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    if homography_matrix is None:
        return None

    yaw, pitch, roll = calculate_yaw_pitch_roll_from_homography(
        homography_matrix)

    # Discard any homography where the image appears to have been reversed or
    # the angle from the camera is too steep.
    # The boundary condition should be `abs(angle) > pi/2` but for some reason
    # the drink was sometimes reporting a "reversed" yaw of 1.9 despite
    # appearing correct in displayed homographies, so experimentally set to 2
    if any((abs(angle) > 2 for angle in (yaw, pitch, roll))):
        if LOG_YAW_PITCH_ROLL:
            print("Dicard detected object due to yaw/pitch/roll angle")
        return None

    if DISPLAY_DETECTED_OBJECTS:
        display_detected_object(query, input_image, good_matches,
                                cluster_input_keypoints, homography_matrix,
                                mask)

    centre_pt, = np.average(dst_pts, axis=0)

    return centre_pt


def detect_objects_in_input_image(input_image):
    input_keypoints, input_descriptors = detector.detectAndCompute(
        input_image, None)

    input_meanshift = create_meanshift(input_keypoints)

    # TODO unused, look up
    # cluster_centers = ms.cluster_centers_

    input_labels = input_meanshift.labels_
    input_labels_unique = np.unique(input_labels)
    input_n_clusters = len(input_labels_unique)
    clustered_descriptor_indexes = [
        np.where(input_meanshift.labels_ == i)[0]
        for i in range(input_n_clusters)
    ]

    if LOG_CLUSTER_MATCHES:
        print(f"number of estimated clusters : {input_n_clusters}\n")

    detected_objects = {}

    for query_name, query in QUERIES.items():
        float_query_descriptors = np.float32(query['descriptors'])

        query_detected_objects = []

        for descriptor_indexes in clustered_descriptor_indexes:
            matches = BF_MATCHER.knnMatch(
                query['descriptors'], input_descriptors[descriptor_indexes,], 2)

            # store all the good_matches matches as per Lowe's ratio test.
            good_matches = [
                m for (m, n) in matches
                if m.distance < LOWE_MATCH_RATIO * n.distance
            ]

            good_matches_count = len(good_matches)
            if good_matches_count >= MIN_MATCH_COUNT:
                if LOG_CLUSTER_MATCHES:
                    print(f"{query_name} - Match: "
                          f"{len(good_matches)}/{MIN_MATCH_COUNT}")

                detected_object_centre = locate_detected_object_centre(
                    query, input_image, input_keypoints, input_descriptors,
                    good_matches, descriptor_indexes)

                if detected_object_centre is not None:
                    query_detected_objects.append(
                        (good_matches_count, detected_object_centre))
                    print(f"{query_name} - Match at {detected_object_centre}: "
                          f"{len(good_matches)}/{MIN_MATCH_COUNT}")
                else:
                    print(f"{query_name} - Match, but no homography "
                          f"{len(good_matches)}/{MIN_MATCH_COUNT}")
            else:
                if LOG_CLUSTER_MATCHES:
                    print(f"{query_name} - No Match: "
                          f"{len(good_matches)}/{MIN_MATCH_COUNT}")

        if query_detected_objects:
            detected_objects[query_name] = tuple(
                sorted(query_detected_objects, key=lambda qdo: qdo[0]))
        else:
            print(f"{query_name} - No Match")

    return detected_objects


def run_test_match_objects():
    for img_name, expected_object_counts in TEST_IMG_MATCHES.items():
        input_img = cv.imread(f'imgs/test/{img_name}.png', 0)
        detected_objects = detect_objects_in_input_image(input_img)

        actual_object_counts = {
            object_name: len(detections)
            for object_name, detections in detected_objects.items()
        }

        assert expected_object_counts == actual_object_counts, \
            f"""test img {img_name}:
            expected {expected_object_counts}
            actual   {actual_object_counts}"""


def run_test_match_written_screenshot():
    input_image = cv.imread('imgs/screen.png', 0)
    detected_objects = detect_objects_in_input_image(input_image)

    print(f"Matched {tuple(detected_objects.keys())}")


def run_bot_service():
    while True:
        input_image = pyautogui.screenshot()
        input_image = cv.cvtColor(np.array(input_image), cv.COLOR_RGB2BGR)
        cv.imwrite('imgs/screen.png', input_image)

        try:
            detected_objects = detect_objects_in_input_image(input_image)
        except cv.error as e:
            print(f"Skipping Frame: Caught cv.error {e}")
            continue
        except Exception as e:
            print(f"Skipping Frame: Caught exception {e}")
            continue

        print(f"Matched {tuple(detected_objects.keys())}")

        sleep(1)


if __name__ == '__main__':
    COMMANDS = {
        'test': run_test_match_objects,
        'test_screen': run_test_match_written_screenshot,
        'bot': run_bot_service
    }

    COMMANDS[sys.argv[1]]()
