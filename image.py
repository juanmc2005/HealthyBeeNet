import cv2

image = cv2.imread("bee_dataset/bee_imgs/001_044.png")
color = cv2.cvtColor(image, cv2.IMREAD_COLOR)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(color, None)
cv2.drawKeypoints(color, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', image)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
print(descs)
