# coding: utf-8

# Useful links
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects

import cv
from matplotlib import pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
from sklearn.cluster import MeanShift, estimate_bandwidth

MIN_MATCH_COUNT = 5
LOWE_MATCH_RATIO = 0.7

orb = cv.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)

QUERY_NAMES = ('sburg', 'dburg', 'fburg', 'fries', 'drink', 'done')
QUERIES = {
    query_name: {
        'name': query_name,
        'image': cv.imread(f'{query_name}.png', 0)
    }
    for query_name in QUERY_NAMES
}
for query_name, query in QUERIES.items():
    query_keypoints, query_descriptors = orb.detectAndCompute(
        query['image'], None)
    query.update(keypoints=query_keypoints, descriptors=query_descriptors)

img2 = cv.imread('testimgs/1.png', 0)  # trainImage


def create_meanshift(keypoints):
    x = np.array([keypoints[0].pt])

    for i in range(len(keypoints)):
        x = np.append(x, [keypoints[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    return ms


def plot_matches(good_matches, query, train_keypoints):
    src_pts = np.float32([
        query['keypoints'][m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)
    dst_pts = np.float32([train_keypoints[m.trainIdx].pt
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

    img2_poly = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)

    print(
        f'Match for {query["name"]} at (): {len(good_matches)}/{MIN_MATCH_COUNT}'
    )

    draw = False
    if not draw:
        return

    img3 = cv.drawMatches(query['image'], query['keypoints'], img2_poly,
                          train_keypoints, good_matches, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()


def find_features_in_train_image(train_image):
    train_keypoints, train_descriptors = orb.detectAndCompute(train_image, None)

    meanshift = create_meanshift(train_keypoints)

    # TODO unused, look up
    # cluster_centers = ms.cluster_centers_

    labels = meanshift.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    s = [None] * n_clusters_
    for i in range(n_clusters_):
        d, = np.where(meanshift.labels_ == i)
        print(d.__len__())
        s[i] = list(train_keypoints[xx] for xx in d)

    des2_ = train_descriptors

    for _, query in QUERIES.items():
        for i in range(n_clusters_):

            train_keypoints = s[i]
            d, = np.where(meanshift.labels_ == i)
            train_descriptors = des2_[d,]

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv.FlannBasedMatcher(index_params, search_params)

            query_descriptors = np.float32(query['descriptors'])
            train_descriptors = np.float32(train_descriptors)

            matches = flann.knnMatch(query_descriptors, train_descriptors, 2)

            # store all the good_matches matches as per Lowe's ratio test.
            good_matches = [
                m for (m, n) in matches
                if m.distance < LOWE_MATCH_RATIO * n.distance
            ]

            if len(good_matches) > MIN_MATCH_COUNT:
                plot_matches(good_matches, query, train_keypoints)
            else:
                print("Not enough matches are found - %d/%d" %
                      (len(good_matches), MIN_MATCH_COUNT))
                matchesMask = None


find_features_in_train_image(img2)
