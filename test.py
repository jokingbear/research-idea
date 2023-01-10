import time

def check(d):
    print('start process')
    while True:
        time.sleep(1)
        if d[0]:
            print('haha')
            return
