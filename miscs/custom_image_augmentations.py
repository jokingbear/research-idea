import numpy as np

from albumentations import Lambda


def cut_out_aug(hr=0.05, wr=0.05, p=0.5):

    def cut_out(img, **kwargs):
        if hr == 0 or wr == 0:
            return img

        shape = img.shape

        h, w = shape[:2]
        pad_h = int(h * hr)
        pad_w = int(w * wr)

        center_x = np.random.randint(pad_h, h - pad_h + 1)
        center_y = np.random.randint(pad_w, h - pad_w + 1)

        mask = np.ones([h, w, 1] if len(shape) == 3 else [h, w])

        mask[..., center_x-pad_h:center_x+pad_h, center_y-pad_w:center_y+pad_w] = 0

        return mask * img

    return Lambda(image=cut_out, mask=cut_out, p=p)
