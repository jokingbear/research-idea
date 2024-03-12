import multiprocessing as mp


def haha(i, q):
    while True:
        print('thread ', i, q)
        if len(q) == 5:
            break


if __name__ == '__main__':
    q = 2
    print(mp.JoinableQueue.__qualname__)