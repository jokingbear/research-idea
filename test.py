import plasma.logging as logger
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@logger.Timer()
@logger.FunctionLogger()
def haha(x, y):
    return x + y


class A:

    @logger.FunctionLogger()
    def run(self, x, y):
        return x + y

a = A()
haha(5, 6)
a.run(5, 6)
