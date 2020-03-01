import multiprocessing
import os
import sys
import traceback
from .. import queues

here = os.path.abspath(os.path.dirname(__file__))

# 将源码根目录加入path， 可以直接引入utils里面的方法, algorithm模块里的算法
source_path = os.path.join(here, '../..')
if source_path not in sys.path:
    sys.path.append(source_path)

from utils.decorator import UnitTestDecorator

# TODO add join method
@UnitTestDecorator
class BaseNode(object):
    '''
    系统节点
    '''

    def __init__(self, process_size=0, queue_type="ProcessingQueue", is_in=True, is_out=True):
        """
            process_size (int, optional): Defaults to 0. 此节点所需要开启的进程数，如果为零则直接在当前进程运行
            queue_type (str, optional): Defaults to "ProcessingQueue". 指定进程间队列类型。"ProcessingQueue": python 原生进程队列。"RedisQueue": Redis队列。
        """

        # q_in是输入队列，q2是输出结果队列
        self.q_in = queues.Queue.get_by_name(
            queue_type, "store_data") if is_in else None
        self.q_out = queues.Queue.get_by_name(
            queue_type, "store_result") if is_out else None
        self.process_size = process_size

    def _run_sigle_process(self, i):
        """多进程运行的接口，为该节点启动一个进程

        Args:
            i (int): 进程编号

        Raises:
            NotImplementedError: 子类必须重写该方法
        """

        raise NotImplementedError

    def _run_single_process_debug_wrapper(self, *args, **kwargs):
        """多进程运行时程序报错打印错误信息
        """

        try:
            self._run_sigle_process(*args, **kwargs)
        except:
            traceback.print_exc()

    def put(self, obj):
        if self.q_in is not None:
            self.q_in.put(obj)
        else:
            print("Warning. This node '%s' has no 'q_in' and it can't recieve message from other nodes." %
                  self.__class__.__name__)

    def get(self):
        if self.q_out is not None:
            return self.q_out.get()
        else:
            print("Warning. This node '%s' has no 'q_out' and you can't get any message from this node." %
                  self.__class__.__name__)
            return None

    def init_node(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, type='process'):
        '''开始运行
        '''
        # 在新的进程中分线程运行
        if type == 'threading':
            pool = multiprocessing.Pool(1)
            pool.apply_async(self.run_on_threads)

        # 在当前进程运行
        elif self.process_size == 0:
            self._run_single_process_debug_wrapper(0)

        # 开启子进程运行
        else:
            pool = multiprocessing.Pool(self.process_size)
            for i in range(self.process_size):
                pool.apply_async(
                    self._run_single_process_debug_wrapper, args=(i,))

            pool.close()

    def run_on_threads(self):
        """
        IO 密集型操作可以调用该方法，将主函数分配到多个线程运行。算法节点请勿调用该方法，算法节点请使用正常的 run() 方法
        """
        # 在当前进程运行
        if self.process_size == 0:
            self._run_single_process_debug_wrapper(0)

        # 开启子线程运行
        else:
            pool = multiprocessing.pool.ThreadPool(self.process_size)
            for i in range(self.process_size):
                pool.apply_async(
                    self._run_single_process_debug_wrapper, args=(i,))

            pool.close()
