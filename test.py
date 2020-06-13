import cv2
import matplotlib.pyplot as plt

from plasma.training.data import augmentations as augs


crop = augs.MinEdgeCrop(512, always_apply=True)

img = cv2.imread("IMG_9409.jpg")
aug_img = crop(image=img)["image"]

_, axes = plt.subplots(ncols=2)
axes[0].imshow(img)
axes[1].imshow(aug_img)
plt.show()
