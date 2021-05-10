import numpy as np

from scipy.ndimage.interpolation import zoom


def resample(vol, dxyz, new_dxyz, order=1):
    xyz = np.array(vol.shape)
    dxyz = np.array(dxyz)
    new_dxyz = np.array(new_dxyz)

    new_xyz = np.round(xyz * dxyz / new_dxyz)
    size = np.round(new_xyz / xyz)
    new_vol = zoom(vol, size, order=order)

    return new_vol
