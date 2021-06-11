import numpy as np

from scipy.ndimage.interpolation import zoom


def resample(vol, dxyz, new_dxyz, order=3, resample_z=True, z_order=0):
    """
    resample volume with new spacing
    Args:
        vol: volume
        dxyz: old spacing
        new_dxyz: new spacing
        order: interpolation order
        resample_z: whether to resample z seperately
        z_order: interpolation order of z

    Returns: resampled volume
    """
    xyz = np.array(vol.shape)
    dxyz = np.array(dxyz)
    new_dxyz = np.array(new_dxyz)

    new_xyz = np.round(xyz * dxyz / new_dxyz)
    factor = np.round(new_xyz / xyz).tolist()

    if resample_z:
        new_vol = zoom(vol, (factor[0], factor[1], 1), order=order)
        new_vol = zoom(new_vol, (1, 1, factor[-1]), order=z_order)
    else:
        new_vol = zoom(vol, tuple(factor), order=order)

    return new_vol


def crop_or_pad(vol, axis, size):
    """
    crop or pad a volume
    Args:
        vol: volume to be cropped or padded
        axis: axis to be cropped or padded along
        size: the size of final volume

    Returns: cropped or padded volume
    """
    if not isinstance(axis, int) or not isinstance(size, int):
        new_vol = vol
        for a, s in zip(axis, size):
            new_vol = crop_or_pad(new_vol, a, s)

        return new_vol

    if axis < 0:
        axis = len(vol.shape) - axis
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
