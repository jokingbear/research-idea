import csv

from keras import callbacks as cbks


class IterationLogger(cbks.Callback):
    def __init__(self, file_path):
        self.csv = open(file_path, "w", newline="")
        self.writer = None
        
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv, logs.keys())
            self.writer.writeheader()

        self.writer.writerow(logs)

    def on_train_end(self, logs=None):
        self.csv.close()
        self.writer = None
