from albumentations import ShiftScaleRotate

aug = ShiftScaleRotate(rotate_limit=15, always_apply=True)
