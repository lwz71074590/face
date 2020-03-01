"""
FIFO queue set for multithread, multiprocessing, redis
"""
import sys
import pickle


class Queue(object):

    def put(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def qsize(self):
        raise NotImplementedError

    @staticmethod
    def get_by_name(name, *args, **kwargs):
        """Get Queue instance implement for multithread, multiprocessing or redis

        Args:
            name (str): one of "ThreadQueue", "ProcessingQueue", "RedisQueue"
        """
        this_module = sys.modules[__name__]
        if hasattr(this_module, name):
            return getattr(this_module, name)(*args, **kwargs)

        else:
            raise AttributeError("No queue named %s." % name)


class ThreadQueue(Queue):
    """FIFO Queue for multithread.
    """

    def __init__(self, tag):
        import queue as q

        self.tag = tag
        self.queue = q.Queue()

    def put(self, object):
        self.queue.put(object)

    def get(self):
        return self.queue.get()

    def qsize(self):
        return self.queue._qsize()


class ProcessingQueue(Queue):
    """FIFO Queue for multiprocess.
    """

    def __init__(self, tag):
        import multiprocessing

        self.tag = tag
        self.queue = multiprocessing.Manager().Queue()

    def put(self, object):
        self.queue.put(object)

    def get(self):
        return self.queue.get()

    def qsize(self):
        return self.queue.qsize()


class RedisQueue(Queue):
    """FIFO based on redis.
    """


    default_connection = dict(
        host='localhost', port=6379, db=0
    )

    def __init__(self, tag):
        import redis

        self.tag = tag
        self.client = redis.StrictRedis(**self.default_connection)

    def put(self, object, c_id=None):
        obj_str = pickle.dumps(object)
        if c_id is None:
            self.client.lpush(self.tag, obj_str)
        else:
            self.client.lpush(self.tag + '_' + str(c_id), obj_str)

    def get(self, c_id=None):
        if c_id is None:
            _, obj_str = self.client.brpop(self.tag)
        else:
            _, obj_str = self.client.brpop(self.tag + '_' + str(c_id))
        obj = pickle.loads(obj_str)
        return obj

    def qsize(self, c_id=None):
        if c_id is None:
            return self.client.llen(self.tag)
        else:
            return self.client.llen(self.tag + '_' + str(c_id))