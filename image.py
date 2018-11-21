import cv2

image = cv2.imread("bee_dataset/bee_imgs/001_043.png")
gray = cv2.cvtColor(image, cv2.IMREAD_COLOR)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
print(descs)
