from tqdm import tqdm
import time
import threading

# 定义处理函数
def process(i):
    for j in tqdm(range(100), desc='Thread {}'.format(i)):
        time.sleep(0.01)

# 创建多个线程
threads = [threading.Thread(target=process, args=(i,)) for i in range(4)]

# 启动线程
for t in threads:
    t.start()

# 等待线程结束
for t in threads:
    t.join()
