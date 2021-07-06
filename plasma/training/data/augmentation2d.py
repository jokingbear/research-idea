import cv2
import numpy as np
import pandas as pd

from albumentations import DualTransform, ImageOnlyTransform


class MinEdgeCrop(DualTransform):
    _default_positions = pd.Index(data=["center", "left", "right"])

    def __init__(self, positions=None, always_apply=True, p=0.5):
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

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, always_apply=True, p=0.5):
        """
        :param size: final size of min edge
        :param interpolation: how to interpolate image
        :param always_apply:
        :param p:
        """
        super().__init__(always_apply, p)

        self.size = size
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        """
        resize image based on its min edge
        :param img: image to be resized
        :param interpolation: how to interpolate an image, value according to cv2.INTER_*
        :param params: not used
        :return: resized image
        """
        h, w = img.shape[:2]
        min_edge = min(h, w)

        size = self.size
        new_h = np.round(h / min_edge * size).astype(int)
        new_w = np.round(w / min_edge * size).astype(int)
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        return img

    def get_params(self):
        return {
            "interpolation": self.interpolation
        }


class ToTorch(DualTransform):

    def __init__(self):
        super().__init__(always_apply=True)

    def apply(self, img, **params):
        assert len(img.shape) in {2, 3}, f"image shape must either be (h, w) or (h, w, c), currently {img.shape}"

        if len(img.shape) == 2:
            return img[np.newaxis]
        else:
            return img.transpose([2, 0, 1])


class HistogramMatching(ImageOnlyTransform):

    def __init__(self, ref_hist, p=1, always_apply=False):
        super().__init__(always_apply, p)

        self.ref_cdf = self._get_cdf_(ref_hist)

    def apply(self, img, **params):
        vals, counts = np.unique(img, return_counts=True)
        size = img.shape[0] * img.shape[1]

        probs = counts / size
        probs = {v: p for v, p in zip(vals, probs)}
        probs = [probs.get(i, 0) for i in range(256)]

        cdf = self._get_cdf_(probs)

        mapping = self._map_cdf_(cdf, self.ref_cdf)
        return mapping[img]

    def _get_cdf_(self, hist):
        cdf = []
        acc = 0

        for i in range(256):
            acc += hist[i]
            cdf.append(acc)

        return np.array(cdf)

    def _map_cdf_(self, in_cdf, ref_cdf):
        mapping = []
        for i in range(256):
            val = in_cdf[i]

            start = np.where(ref_cdf <= val)[0]
            if len(start) == 0:
                start = 0
            else:
                start = start[-1]

            end = np.where(val <= ref_cdf)[0]
            if len(end) == 0:
                end = 255
            else:
                end = end[0]

            start_val = ref_cdf[start]
            end_val = ref_cdf[end]

            if val < start_val:
                mapping.append(start)
            elif abs(start_val - val) < abs(val - end_val):
                mapping.append(start)
            else:
                mapping.append(end)

        return np.array(mapping)
