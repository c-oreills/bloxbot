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

img1 = cv.imread('sburg.png', 0)  # queryImage
img2 = cv.imread('testimgs/1.png', 0)  # trainImage

orb = cv.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)

# find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


def create_meanshift(keypoints):
    x = np.array([keypoints[0].pt])

    for i in range(len(keypoints)):
        x = np.append(x, [keypoints[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    return ms


def plot_matches(good_matches, keypoints1, keypoints2):
    src_pts = np.float32([keypoints1[m.queryIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)

    if M is None:
        print("No Homography")
    else:
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img2_poly = cv.polylines(img2, [np.int32(dst)], True, 255, 3,
                                 cv.LINE_AA)

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2)

        img3 = cv.drawMatches(img1, keypoints1, img2_poly, keypoints2,
                              good_matches, None, **draw_params)

        plt.imshow(img3, 'gray'), plt.show()


meanshift = create_meanshift(keypoints2)

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
    s[i] = list(keypoints2[xx] for xx in d)

des2_ = descriptors2

for i in range(n_clusters_):

    keypoints2 = s[i]
    d, = np.where(meanshift.labels_ == i)
    descriptors2 = des2_[d,]

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    matches = flann.knnMatch(descriptors1, descriptors2, 2)

    # store all the good_matches matches as per Lowe's ratio test.
    good_matches = [
        m for (m, n) in matches if m.distance < LOWE_MATCH_RATIO * n.distance
    ]

    if len(good_matches) > MIN_MATCH_COUNT:
        plot_matches(good_matches, keypoints1, keypoints2)
    else:
        print("Not enough matches are found - %d/%d" %
              (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None
