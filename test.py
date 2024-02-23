import plasma.logging as logger
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@logger.Timer()
@logger.FunctionLogger()
def haha(x):
    return x


class A:

    @logger.Timer()
    @logger.FunctionLogger()
    def run(self, x):
        return x

a = A()
haha('string')
a.run('string')
