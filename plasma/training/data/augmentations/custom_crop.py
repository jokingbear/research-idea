import cv2
import numpy as np

from albumentations import DualTransform


class MinEdgeCrop(DualTransform):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.size = size
        self.interpolation = interpolation

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("no implementation for bbox")

    def apply(self, img, **params):
        h, w = img.shape[:2]

        assert h % 2 == 0 and w % 2 == 0, "height and width must be even"

        min_edge = min(h, w)
        position = params["position"]
        if h > min_edge:
            if position == "left":
                img = img[:min_edge]
            elif position == "center":
                d = (h - min_edge) // 2
                img = img[d:-d]
            else:
                img = img[-min_edge:]

        if w > min_edge:
            if position == "left":
                img = img[:, :min_edge]
            elif position == "center":
                d = (w - min_edge) // 2
                img = img[:, d:-d]
            else:
                img = img[:, -min_edge:]

        return cv2.resize(img, (self.size, self.size), interpolation=self.interpolation)

    def get_params(self):
        return {
            "position": np.random.choice(["left", "center", "right"])
        }

    def get_params_dependent_on_targets(self, params):
        pass
