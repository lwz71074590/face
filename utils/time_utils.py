'''
@Author: TangZhiFeng
@Data: 2019-01-03
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-03 16:50:33
@Description: 时间的工具
'''



def genera_stamp():
    '''生成时间戳
    
    Returns:
        str -- 字符型的时间戳
    '''

    import time
    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(now/1000))
    return now02.__str__()