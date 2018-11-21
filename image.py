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
    _, descriptors = op.detectAndCompute(color, None)
    return descriptors


def draw_key_points(image, out_image):
    """
    Applies the SIFT algorithm to the input image and draws them in the output image
    :param img: the source image path to look for key points
    :param out_image: the output image path to draw the key points
    :return: nothing
    """
    img = cv2.imread(image)
    color = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    op = cv2.xfeatures2d.SIFT_create()
    kps, descriptors = op.detectAndCompute(color, None)
    cv2.drawKeypoints(color, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_image, img)
