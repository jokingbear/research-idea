import pickle


def obj2pickle(obj, path):
    with open(path, 'wb') as handler:
        pickle.dump(obj, handler)
