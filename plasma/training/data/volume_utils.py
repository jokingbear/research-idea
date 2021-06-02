import numpy as np

from scipy.ndimage.interpolation import zoom


def resample(vol, dxyz, new_dxyz, order=3):
    xyz = np.array(vol.shape)
    dxyz = np.array(dxyz)
    new_dxyz = np.array(new_dxyz)

    new_xyz = np.round(xyz * dxyz / new_dxyz)
    factor = np.round(new_xyz / xyz).tolist()
    new_vol = zoom(vol, tuple(factor), order=order)

    return new_vol


def crop_or_pad(vol, axis, size):
    original_size = vol.shape[axis]

    total_diff = original_size - size
    if total_diff == 0:
        return vol

    diff = abs(total_diff) // 2
    left = diff
    right = (diff + 1) if total_diff % 2 != 0 else diff

    if total_diff > 0:
        slices = [slice(None, None) if a != axis else slice(left, -right) for a, _ in enumerate(vol.shape)]
        return vol[tuple(slices)]
    else:
        pad = [(0, 0) if a != axis else (left, right) for a, _ in enumerate(vol.shape)]
        return np.pad(vol, pad, constant_values=vol.min())
