import math
import sys

import cv2
# import imutils
import numpy as np


def decompose_2d_matrix(mat):
    print(mat)
    theta0 = np.degrees(np.arctan2(-mat[0, 1], mat[0, 0]))
    print(theta0)
    a = mat[0][0]
    b = mat[0][1]
    c = mat[0][2]
    d = mat[1][0]
    e = mat[1][1]
    f = mat[1][2]

    delta = a * d - b * c

    translation = [e, f]
    rotation = 0
    scale = [0, 0]
    skew = [0, 0]

    # Apply the QR-like decomposition.
    if a != 0 or b != 0:
        r = math.sqrt(a * a + b * b)
        rotation = math.acos(a / r) if b > 0 else -math.acos(a / r)
        scale = [r, delta / r]
        skew = [math.atan((a * c + b * d) / (r * r)), 0]
    elif c != 0 or d != 0:
        s = math.sqrt(c * c + d * d)
        rotation = math.PI / 2 - (math.acos(-c / s) if d > 0 else -math.acos(c / s))
        scale = [delta / s, s]
        skew = [0, math.atan((a * c + b * d) / (s * s))]
    else:
        # a = b = c = d = 0
        pass

    print("----")
    print(translation)
    # print(rotation)
    d = math.degrees(rotation)
    print(d, -d)
    # print(scale)
    l = math.sqrt(scale[0] ** 2 + scale[1] ** 2)
    print(l, (1 / l))
    # print(skew)
    print("----")


def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    # if debug:
    #     matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
    #         matches, None)
    #     matchedVis = imutils.resize(matchedVis, width=1000)
    #     cv2.imshow("Matched Keypoints", matchedVis)
    #     cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    #(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    #print(H)

    (M, _) = cv2.estimateAffinePartial2D(ptsA, ptsB)

    print("####")
    print(template.shape[:2])
    decompose_2d_matrix(M)
    print("####")

    sys.exit("foobar")

    # Fake the homography matrix from our 2D matrix
    # H = np.array([M[0], M[1], [0, 0, 1]], dtype=float)

    print(M)
    #print(H)
    src = np.array([[[0, 0]]], dtype=float)
    dst = cv2.transform(src, M)
    print(dst[0][0])
    src = np.array([[[1, 0]]], dtype=float)
    dst = cv2.transform(src, M)
    xyz2x = dst[0][0][0]
    xyz2y = dst[0][0][1]
    print(xyz2x, xyz2y)
    print(math.sqrt(xyz2x ** 2 + xyz2y ** 2))
    rotate = math.degrees(math.atan2(xyz2y, xyz2x))
    print(rotate)
    print("GCH1")
    # // extract translation
    # var p:Point = new Point();
    # translate = m.transformPoint(p);
    # m.translate( -translate.x, -translate.y);
    #
    # // extract (uniform) scale
    # p.x = 1.0;
    # p.y = 0.0;
    # p = m.transformPoint(p);
    # scale = p.length;
    #
    # // and rotation
    # rotate = Math.atan2(p.y, p.x);

    sys.exit("foobar")

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned
