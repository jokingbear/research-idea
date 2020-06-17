import numpy as np

from albumentations import DualTransform


class MinEdgeCrop(DualTransform):

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("no implementation for bbox")

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("no implementation for bbox")

    def apply(self, img, position="center"):
        h, w = img.shape[:2]

        assert h % 2 == 0 and w % 2 == 0, "height and width must be even"

        min_edge = min(h, w)
        if h > min_edge:
            if position == "left":
                img = img[:min_edge]
            elif position == "center":
                d = (h - min_edge) // 2
                img = img[d:-d]

                if h % 2 != 0:
                    img = img[1:]
            else:
                img = img[-min_edge:]

        if w > min_edge:
            if position == "left":
                img = img[:, :min_edge]
            elif position == "center":
                d = (w - min_edge) // 2
                img = img[:, d:-d]

                if w % 2 != 0:
                    img = img[:, 1:]
            else:
                img = img[:, -min_edge:]

        assert img.shape[0] == img.shape[1], f"height and width must be the same, currently {img.shape[:2]}"
        return img

    def get_params(self):
        return {
            "position": np.random.choice(["left", "center", "right"])
        }

    def get_params_dependent_on_targets(self, params):
        pass
