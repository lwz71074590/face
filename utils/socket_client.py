'''
@Author: TangZhiFeng
@Data: 2018-12-28 15:50:58 
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-04 15:59:16
@Description: 向socket service端发送请求
'''
import websocket
try:
    import thread
except ImportError:
    import _thread as thread
import time
import queue

q = queue.Queue()


def on_message(ws, message):
    '''接受Service返回的数据，当前作用不大

    Arguments:
        ws {websocket} -- 该连接对象
        message {str} -- 返回的文本内容
    '''

    pass


def on_error(ws, error):
    '''发送错误，需要写入日志。TODO

    Arguments:
        ws {websocket} -- 连接对象
        error {exception} -- 返回的Exception对象
    '''
    pass


def on_close(ws):
    '''断开连接

    Arguments:
        ws {websocket} -- 连接对象
    '''
    pass


def on_open(ws):
    def run(*args):
        try:
            print('connecting success')
            while True:
                msg = q.get()
                ws.send(msg)
            print("thread terminating...")
        except Exception as e:
            import traceback
            traceback.print_exc()
    thread.start_new_thread(run, ())


class Client(object):
    def __init__(self, host):
        '''WebSocket client 初始化

        Arguments:
            host {str} -- 请求地址
        '''
        self.ws = None
        self.connect(host)

    def connect(self, host):
        '''连接摄像头操作

        Arguments:
            host {str} -- Websocket服务地址
        '''

        def run():
            self.ws = websocket.WebSocketApp(host,
                                             on_message=on_message,
                                             on_error=on_error,
                                             on_close=on_close)
            self.ws.on_open = on_open
            result = self.ws.run_forever()
            if result is True:
                print('iother exception was raised during a loop')
            else:
                print('caught KeyboardInterrupt')
        thread.start_new_thread(run, ())

    def send(self, msg):
        '''向Websocket Service发送数据

        Arguments:
            msg {str} -- 文本的内容，一般情况是json格式的数据
        '''

        q.put(msg)

    def close(self):
        self.ws.close()

client = Client('ws://192.168.0.140:9501')

