import cv2


def sift(image):
    """
    Applies the SIFT algorithm to an image
    :param image: the path to a bee image
    :return: a matrix of shape (k, 128), where k is the number
        of key points detected by SIFT, with the descriptors for every key point
    """
    image = cv2.imread(image)
    color = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    op = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = op.detectAndCompute(color, None)
    return descriptors


# Draw key points to inspect image TODO make a function
# cv2.drawKeypoints(color, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg', image)
