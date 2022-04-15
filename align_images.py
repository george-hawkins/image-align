import argparse
import math
import sys

import cv2
import numpy as np


_MAX_FEATURES = 500
_KEEP_PERCENT = 0.2


def _vertical_flip(height, pts1, pts2):
    top = height - 1
    for i in range(len(pts1)):
        pts1[i, 1] = top - pts1[i, 1]
        pts2[i, 1] = top - pts2[i, 1]


def _origin_to_center(height, width, pts1, pts2):
    x_offset = width / 2
    y_offset = height / 2

    for i in range(len(pts1)):
        pts1[i, 0] -= x_offset
        pts1[i, 1] -= y_offset
        pts2[i, 0] -= x_offset
        pts2[i, 1] -= y_offset


# Adjust the coordinates to match the system used by Blender.
# TODO: surely it would much cheaper to do these transformations _after_ estimating the similarity matrix.
def _blender_coordinates(height, width, pts1, pts2):
    # Flip so top become bottom.
    _vertical_flip(height, pts1, pts2)

    # Move the origin to the center to the image.
    _origin_to_center(height, width, pts1, pts2)


def align_images(orig_img, transformed_img):
    # Convert both the original and transformed images grayscale.
    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    transformed_gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)

    # Use ORB to detect key points and extract (binary) local invariant features.
    orb = cv2.ORB_create(_MAX_FEATURES)
    (kp1, des1) = orb.detectAndCompute(orig_gray, None)
    (kp2, des2) = orb.detectAndCompute(transformed_gray, None)

    # Match the features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    # Sort the matches by their distance (the smaller the distance, # the "more similar" the features are).
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * _KEEP_PERCENT)
    matches = matches[:keep]

    # Print the worst distance that will be used, 30 or above is poor.
    worst_distance = matches[-1].distance
    print(f"Worst match distance: {worst_distance}")

    # Extract the best matching points into two arrays of points.
    pts1 = np.empty((keep, 2), dtype="float")
    pts2 = np.empty((keep, 2), dtype="float")
    for (i, m) in enumerate(matches):
        # Indicate that the two key points in the respective images map to each other.
        pts1[i] = kp1[m.queryIdx].pt
        pts2[i] = kp2[m.trainIdx].pt

    height = orig_img.shape[0]
    width = orig_img.shape[1]
    print(f"Dimensions: {width}x{height}")

    _blender_coordinates(height, width, pts1, pts2)

    # We want a similarity matrix (with four degrees of freedom) rather than a general affine matrix (with
    # six degrees of freedom) so that we can decompose it into just scale, rotation and translation. Hence,
    # the use of `estimateAffinePartial2D` rather than `estimateAffine2D`.
    (M, _) = cv2.estimateAffinePartial2D(pts1, pts2)

    t_x = M[0, 2]
    t_y = M[1, 2]
    theta = math.atan2(-M[0, 1], M[0, 0])

    # Is there a better rule for choosing between M[0, 0] and M[1, 0] for calculating scale?
    # At the moment, I chose the one where the trig result will be furthest from 0.
    if (math.pi / 4) < abs(theta) < (math.pi * 3 / 4):
        scale = M[1, 0] / math.sin(theta)
    else:
        scale = M[0, 0] / math.cos(theta)

    # I can't see where this issue is creeping in but adding 0.5 to x and y gets closer to the desired values.
    t_x += 0.5
    t_y += 0.5

    # Output values to use in Blender Transform node.
    # Blender views any value smaller than 0.000001 as 0 hence the use of .6f.
    print()
    print("Transform node values:")
    print(f"* X: {t_x:.6f}")
    print(f"* Y: {t_y:.6f}")
    print(f"* Angle: {math.degrees(theta):.6f}")
    print(f"* Scale: {scale:.6f}")


def load(filename):
    img = cv2.imread(filename)
    if img is None:
        sys.exit(f'Error: could not load "{filename}\"')
    return img


# Usage: --original orig-variations/ra-xx.png --transformed variations5/ra-xx.png
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True, help="path to original image")
    ap.add_argument("-t", "--transformed", required=True, help="path to transformed image")
    args = vars(ap.parse_args())

    orig_img = load(args["original"])
    transformed_img = load(args["transformed"])
    align_images(orig_img, transformed_img)


if __name__ == '__main__':
    main()
