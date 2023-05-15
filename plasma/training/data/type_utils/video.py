import cv2
import numpy as np


class Video:
    def __init__(self, video_path):
        self.vid = cv2.VideoCapture(video_path)
        start_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        total_frame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = total_frame - start_frame

    @property
    def fps(self):
        return self.vid.get(cv2.CAP_PROP_FPS)

    @property
    def duration(self):
        return self.nframes / self.fps

    @property
    def shape(self):
        h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self.nframes, h, w, 3

    def subclip(self, start_t, end_t):
        assert 0 <= start_t < end_t < self.duration, "recheck `start_t` and `end_t`"

        start_frame = np.round(start_t * self.fps).astype(int)
        end_frame = np.round(end_t * self.fps).astype(int)

        self.nframes = end_frame - start_frame
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        return self

    def iter_frames(self):
        start = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        for frame_idx in range(self.nframes):
            success, image = self.vid.read()

            if not success:
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, start)
                break

            yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, start)
