'''子算法基类，后面被用于scenes下
'''

import multiprocessing
from . import queues


class Algorithm(object):
    '''
    子算法模块
    '''

    def __init__(self, algo, real_time=True, process_size=None):
        self.algo = algo
        # q1是需要处理的数据，q2是处理完了的数据
        self.q1 = queues.Queue.get_by_name(
            "ProcessingQueue", "store_data")
        self.q2 = queues.Queue.get_by_name(
            "ProcessingQueue", "store_result")
        self.process_size = process_size
        if real_time:
            self.run()

    def __run_multiprocessing(self):
        '''多进程运行的接口，每一个算法如果有多进程的功能需要重写这个函数

        Arguments:
            process_size {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''

        raise NotImplementedError('__run_multiprocessing')

    def __run_nonmultiprocessing(self):
        '''非多进程运行（在主线程中运行）

        Raises:
            NotImplementedError -- 需要重写的Exception
        '''

        raise NotImplementedError('__run_nonmultiprocessing')

    def run(self):
        '''开始运行
        '''
        if self.process_size is None:
            self.__run_nonmultiprocessing()
        else:
            self.__run_multiprocessing()