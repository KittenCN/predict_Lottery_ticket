# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import argparse
import datetime
from config import *
from loguru import logger
from common import setMiniargs, get_current_number, run_predict, predict_run,init, red_graph, blue_graph, pred_key_d, red_sess, blue_sess
from common import tf as predict_tf
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择训练数据")
parser.add_argument('--windows_size', default='3', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--cq', default=1, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
args = parser.parse_args()

if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        init()
        setMiniargs(args)
        list_windows_size = args.windows_size.split(",")
        if list_windows_size[0] == "-1":
            list_windows_size = []
            path = model_path + model_args[args.name]["pathname"]['name']
            dbtype_list = os.listdir(path)
            for dbtype in dbtype_list:
                try:
                    list_windows_size.append(int(dbtype))
                except:
                    pass
            if len(list_windows_size) == 0:
                raise Exception("没有找到训练模型！")
            list_windows_size.sort(reverse=True)   
            logger.info(path)
            logger.info("windows_size: {}".format(list_windows_size))
        for size in list_windows_size:
            predict_tf.compat.v1.reset_default_graph()
            red_graph = predict_tf.compat.v1.Graph()
            blue_graph = predict_tf.compat.v1.Graph()
            pred_key_d = {}
            red_sess = predict_tf.compat.v1.Session(graph=red_graph)
            blue_sess = predict_tf.compat.v1.Session(graph=blue_graph)
            current_number = get_current_number(args.name)
            run_predict(int(size))
            _data, _title = predict_run(args.name)
        filename = datetime.datetime.now().strftime('%Y%m%d')
        filepath = "{}{}/".format(predict_path, args.name)
        fileadd = "{}{}{}".format(filepath, filename, ".csv")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        df = pd.DataFrame(_data, columns=_title)
        df.to_csv(fileadd, encoding="utf-8",index=False)
        
