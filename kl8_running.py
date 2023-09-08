import subprocess

kl8_analysis = "./kl8_analysis.py"
kl8_cash = "./kl8_cash.py"
cal_nums_list = [4,5,7,10]
total_create_list = [50, 100, 1000]
for _total_create in total_create_list:
    for _cal_nums in cal_nums_list:
        for _current_nums in range(2023140, 2023241):
            subprocess.run(["python", kl8_analysis, "--download", "0", "--total_create", str(_total_create), \
                            "--cal_nums", str(_cal_nums), "--current_nums", str(_current_nums), "--limit_line", "5", \
                            "--path", str(_total_create) + '_' + str(_cal_nums), "--repeat", "5", "--simple_mode", "1"])