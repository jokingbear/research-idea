import cv2
import numpy as np
import pandas as pd

from albumentations import DualTransform


class MinEdgeCrop(DualTransform):
    _default_positions = pd.Index(data=["center", "left", "right"])

    def __init__(self, positions=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.positions = positions or self._default_positions

    def apply(self, img, position="center", **kwargs):
        """
        crop image base on min size
        :param img: image to be cropped
        :param position: where to crop the image
        :return: cropped image
        """
        assert position in self._default_positions, "position must either be: left, center or right"

        h, w = img.shape[:2]

        if h == w:
            return img

        min_edge = min(h, w)
        if h > min_edge:
            if position == "left":
                img = img[:min_edge]
            elif position == "center":
                d = (h - min_edge) // 2
                img = img[d:-d] if d != 0 else img

                if h % 2 != 0:
                    img = img[1:]
            else:
                img = img[-min_edge:]

        if w > min_edge:
            if position == "left":
                img = img[:, :min_edge]
            elif position == "center":
                d = (w - min_edge) // 2
                img = img[:, d:-d] if d != 0 else img

                if w % 2 != 0:
                    img = img[:, 1:]
            else:
                img = img[:, -min_edge:]

        assert img.shape[0] == img.shape[1], f"height and width must be the same, currently {img.shape[:2]}"
        return img

    def get_params(self):
        return {
            "position": np.random.choice(self.positions)
        }


class MinEdgeResize(DualTransform):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        """
        :param size: final size of min edge
        :param interpolation: how to interpolate image
        :param always_apply:
        :param p:
        """
        super().__init__(always_apply, p)

        self.size = size
        self.interpolation = interpolation

    def apply(self, img, **params):
        """
        resize image based on its min edge
        :param img: image to be resized
        :param params: not used
        :return: resized image
        """
        h, w = img.shape[:2]
        min_edge = min(h, w)

        size = self.size
        new_h = np.round(h / min_edge * size).astype(int)
        new_w = np.round(w / min_edge * size).astype(int)
        img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

        return img
