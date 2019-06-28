from keras import callbacks as cbks

import pandas as pd


class IterationCallback(cbks.Callback):

    def __init__(self):
        self.frame = pd.DataFrame()

        super().__init__()

    def on_batch_end(self, batch, logs=None):
        df = pd.DataFrame(logs)
        self.frame = self.frame.append(df)
        pass
