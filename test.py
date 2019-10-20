import torch.optim as opts
import cv2

from tensorflow.keras.datasets import mnist


(x_train, _), _ = mnist.load_data()

imgs = x_train / 127.5 - 1
