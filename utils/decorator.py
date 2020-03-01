import traceback
import time
from functools import wraps


def UnitTestDecorator(obj):
    """单元测试装饰器，在单元测试的时候可以开启单元测试模式，不影响程序正常运行时的状态
    """

    def set_test_option_on(self):
        self.test_option = True

    def get_test_option(self):
        return self.test_option

    obj.test_option = False
    obj.set_test_option_on = set_test_option_on
    obj.get_test_option = get_test_option
    return obj


def excution_time(iteration_times):

    def middle(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / iteration_times

            print("Total time cost this function %s is %f." %
                  (func.__name__, total_time))
            print("Avg time cost each opration is %f." % avg_time)
            return ret
        return wrapper
    return middle
