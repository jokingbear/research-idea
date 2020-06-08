import sys
import os


def get_base(path):
    if path[-1] == "/":
        path = path[:-1]

    base = os.path.basename(path)
    path = path.replace(base, "")

    paths = set(sys.path)
    if path not in paths:
        sys.path.append(path)

    return base
