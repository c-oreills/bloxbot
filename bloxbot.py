# coding: utf-8

# Useful links
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects

import sys
from time import sleep

import numpy as np
import pyautogui
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv

# Order images aren't skewed by perspective, so match threshold can be higher
ORDER_MIN_MATCH_COUNT = 30
# Till images get skewed by perspective, so need more room for error
TILL_MIN_MATCH_COUNT = 10
# Till done has very little in the way of features, so needs even more lenience
TILL_DONE_MIN_MATCH_COUNT = 5

LOWE_MATCH_RATIO = 0.7

LOG_CLUSTER_MATCHES = False

orb = cv.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)

QUERY_NAMES = (
    # Order impages (use to parse order)
    'order/sburg',
    'order/dburg',
    'order/fburg',
    'order/fries',
    'order/drink',
    # Till images (use to serve order)
    'till/sburg',
    'till/dburg',
    'till/fburg',
    'till/fries',
    'till/drink',
    'till/done')
QUERIES = {
    query_name: {
        'name': query_name,
        'image': cv.imread(f'imgs/{query_name}.png', 0)
    }
    for query_name in QUERY_NAMES
}
for query_name, query in QUERIES.items():
    query_keypoints, query_descriptors = orb.detectAndCompute(
        query['image'], None)
    query.update(keypoints=query_keypoints, descriptors=query_descriptors)

TEST_IMG_MATCHES = {
    '1': ('order/sburg', 'order/fries', 'till/sburg', 'till/dburg',
          'till/fburg', 'till/fries', 'till/drink', 'till/done'),
    '2': ('order/fburg', 'till/sburg', 'till/dburg', 'till/fburg', 'till/fries',
          'till/drink', 'till/done'),
    '3': (
        'order/fburg',
        'order/fries',
        'order/drink',
        'till/sburg',
        'till/dburg',
        'till/fburg',
        'till/fries',
        'till/drink',
        #'till/done'
    ),
    '4': ('order/sburg', 'order/fries', 'till/sburg', 'till/dburg',
          'till/fburg', 'till/fries', 'till/drink', 'till/done'),
    '5': ('order/dburg', 'order/fries', 'till/sburg', 'till/dburg',
          'till/fburg', 'till/fries', 'till/drink', 'till/done'),
}

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
SEARCH_PARAMS = dict(checks=50)
FLANN_MATCHER = cv.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)


def create_meanshift(keypoints):
    x = np.array([keypoints[0].pt])

    for i in range(len(keypoints)):
        x = np.append(x, [keypoints[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    return ms


def get_min_match_count_for_query(query):
    if query['name'].startswith('order/'):
        return ORDER_MIN_MATCH_COUNT
    if query['name'] == 'till/done':
        return TILL_DONE_MIN_MATCH_COUNT
    if query['name'].startswith('till/'):
        return TILL_MIN_MATCH_COUNT
    raise ValueError('Unknown query type')


def plot_matches(query, input_image, input_keypoints, input_descriptors,
                 good_matches, cluster_descriptor_indexes):

    cluster_input_keypoints = [
        input_keypoints[index] for index in cluster_descriptor_indexes
    ]

    src_pts = np.float32([
        query['keypoints'][m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([
        cluster_input_keypoints[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)

    if M is None:
        print("No Homography")
        return

    matchesMask = mask.ravel().tolist()

    h, w = query['image'].shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    input_img_poly = cv.polylines(input_image, [np.int32(dst)], True, 255, 3,
                                  cv.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)

    centre_pt = np.average(dst_pts, axis=0)
    print(f'Match at {centre_pt}: {len(good_matches)} good matches')

    draw = False
    if not draw:
        return

    match_display_img = cv.drawMatches(query['image'], query['keypoints'],
                                       input_img_poly, cluster_input_keypoints,
                                       good_matches, None, **draw_params)
    plt.imshow(match_display_img, 'gray'), plt.show()


def match_objects_in_input_image(input_image):
    input_keypoints, input_descriptors = orb.detectAndCompute(input_image, None)

    input_meanshift = create_meanshift(input_keypoints)

    # TODO unused, look up
    # cluster_centers = ms.cluster_centers_

    input_labels = input_meanshift.labels_
    input_labels_unique = np.unique(input_labels)
    input_n_clusters = len(input_labels_unique)
    print(f"number of estimated clusters : {input_n_clusters}")
    print()

    object_matches = {}

    for query_name, query in QUERIES.items():
        print(f'# {query_name}')

        min_match_count = get_min_match_count_for_query(query)

        best_match_count, best_match = 0, None

        for i in range(input_n_clusters):
            # "cluster_descriptor_indexes" is a guess as what these actually are
            cluster_descriptor_indexes, = np.where(input_meanshift.labels_ == i)

            matches = FLANN_MATCHER.knnMatch(
                np.float32(query['descriptors']),
                np.float32(input_descriptors[cluster_descriptor_indexes,]), 2)

            # store all the good_matches matches as per Lowe's ratio test.
            good_matches = [
                m for (m, n) in matches
                if m.distance < LOWE_MATCH_RATIO * n.distance
            ]

            good_matches_count = len(good_matches)
            if good_matches_count >= min_match_count:
                if LOG_CLUSTER_MATCHES:
                    print(f"Match: {len(good_matches)}/{min_match_count}")
                if good_matches_count > best_match_count:
                    best_match_count = good_matches_count
                    best_match = (good_matches, cluster_descriptor_indexes)
            else:
                if LOG_CLUSTER_MATCHES:
                    print(f"No Match: {len(good_matches)}/{min_match_count}")

        object_matches[query_name] = best_match

        if best_match:
            good_matches, cluster_descriptor_indexes = best_match
            plot_matches(query, input_image, input_keypoints, input_descriptors,
                         good_matches, cluster_descriptor_indexes)
        else:
            print('No Match')

        print()
    return object_matches


def run_test_match_objects():
    for img_name, expected_match_names in TEST_IMG_MATCHES.items():
        input_img = cv.imread(f'imgs/test/{img_name}.png', 0)
        object_matches = match_objects_in_input_image(input_img)
        actual_match_names = {
            query_name
            for query_name, match in object_matches.items() if match is not None
        }

        assert set(
            expected_match_names
        ) == actual_match_names, f"test img {img_name}; {expected_match_names} != {actual_match_names}"


def run_test_match_written_screenshot():
    input_image = cv.imread('imgs/screen.png', 0)
    object_matches = match_objects_in_input_image(input_image)


def run_bot_service():
    while True:
        input_image = pyautogui.screenshot()
        input_image = cv.cvtColor(np.array(input_image), cv.COLOR_RGB2BGR)
        cv.imwrite('imgs/screen.png', input_image)

        try:
            object_matches = match_objects_in_input_image(input_image)
        except cv.error as e:
            print(f"Skipping Frame: Caught error {e}")
            continue

        successful_matches = {
            query_name
            for query_name, match in object_matches.items() if match is not None
        }

        print(f"Matched {successful_matches}")

        sleep(1)


if __name__ == '__main__':
    COMMANDS = {
        'test': run_test_match_objects,
        'test_screen': run_test_match_written_screenshot,
        'bot': run_bot_service
    }

    COMMANDS[sys.argv[1]]()
