import argparse
import subprocess
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--cal_nums_list', default="4,5,7,10", type=str, help='cal_nums_list')
parser.add_argument('--total_create_list', default="50,100,1000", type=str, help='total_create_list')
parser.add_argument('--nums_range', default="2023140,2023241", type=str, help='nums_range')
args = parser.parse_args()

def _main(_total_create, begin=2023140, end=2023241):
    for _cal_nums in cal_nums_list:
        for _current_nums in range(begin, end):
            subprocess.run(["python", kl8_analysis, "--download", "0", "--total_create", str(_total_create), \
                            "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5", \
                            "--path", str(_total_create) + '_' + str(_cal_nums), "--repeat", "5", "--simple_mode", "1"])


kl8_analysis = "./kl8_analysis.py"
kl8_cash = "./kl8_cash.py"
cal_nums_list = [int(element) for element in args.cal_nums_list.split(',')]
total_create_list = [int(element) for element in args.total_create_list.split(',')]
begin, end = [int(element) for element in args.nums_range.split(',')]
threads = []
for _total_create in total_create_list:
    t = threading.Thread(target=_main, args=(_total_create, begin, end))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
