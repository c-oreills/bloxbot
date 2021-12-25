# coding: utf-8

# Useful links
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects

import cv
from matplotlib import pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
from sklearn.cluster import MeanShift, estimate_bandwidth

MIN_MATCH_COUNT = 10
LOWE_MATCH_RATIO = 0.7

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

input_img = cv.imread('imgs/test/1.png', 0)

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


def plot_matches(good_matches, query, input_keypoints):
    src_pts = np.float32([
        query['keypoints'][m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([input_keypoints[m.trainIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)

    if M is None:
        print("No Homography")
        return

    matchesMask = mask.ravel().tolist()

    h, w = query['image'].shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    img2_poly = cv.polylines(input_img, [np.int32(dst)], True, 255, 3,
                             cv.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)

    centre_pt = np.average(dst_pts, axis=0)
    print(f'Match at {centre_pt}: {len(good_matches)}/{MIN_MATCH_COUNT}')

    draw = False
    if not draw:
        return

    img3 = cv.drawMatches(query['image'], query['keypoints'], img2_poly,
                          input_keypoints, good_matches, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()


def find_features_in_input_image(input_image):
    input_keypoints, input_descriptors = orb.detectAndCompute(input_image, None)

    input_meanshift = create_meanshift(input_keypoints)

    # TODO unused, look up
    # cluster_centers = ms.cluster_centers_

    input_labels = input_meanshift.labels_
    input_labels_unique = np.unique(input_labels)
    input_n_clusters = len(input_labels_unique)
    print(f"number of estimated clusters : {input_n_clusters}")
    print()

    for query_name, query in QUERIES.items():
        print(f'# {query_name}')
        for i in range(input_n_clusters):
            # "descriptor_indexes" is a guess as what these actually are
            descriptor_indexes, = np.where(input_meanshift.labels_ == i)

            matches = FLANN_MATCHER.knnMatch(
                np.float32(query['descriptors']),
                np.float32(input_descriptors[descriptor_indexes,]), 2)

            # store all the good_matches matches as per Lowe's ratio test.
            good_matches = [
                m for (m, n) in matches
                if m.distance < LOWE_MATCH_RATIO * n.distance
            ]

            if len(good_matches) >= MIN_MATCH_COUNT:
                cluster_input_keypoints = [
                    input_keypoints[index] for index in descriptor_indexes
                ]
                plot_matches(good_matches, query, cluster_input_keypoints)
            else:
                print("Not enough matches: %d/%d" %
                      (len(good_matches), MIN_MATCH_COUNT))
                matchesMask = None
        print()


find_features_in_input_image(input_img)
