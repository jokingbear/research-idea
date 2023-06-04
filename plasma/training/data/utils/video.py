import cv2
import numpy as np


class Video:

    def __init__(self, video_path, start_frame=0, duration=None):
        self.video_path = video_path
        self.vid = cv2.VideoCapture(video_path)

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if duration is not None:
            self.vid.set(cv2.CAP_PROP_FRAME_COUNT, duration)

    @property
    def nframes(self):
        return self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

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

    @property
    def resolution(self):
        return self.shape[1:3]

    def subclip(self, start_t, end_t):
        assert 0 <= start_t < end_t < self.duration, "recheck `start_t` and `end_t`"

        start_frame = np.ceil(start_t * self.fps).astype(int)
        end_frame = np.ceil(end_t * self.fps).astype(int)

        return Video(self.video_path, start_frame, end_frame - start_frame)

    def __iter__(self):
        start = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        for frame_idx in range(self.nframes):
            success, image = self.vid.read()

            if not success:
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, start)
                break

            yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, start)
