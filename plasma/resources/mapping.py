import os


def get(path):
    current_path = os.path.dirname(__file__)
    filename = current_path + '/' + path
    return filename
