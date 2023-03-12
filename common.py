# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from config import *
import json
import datetime
import numpy as np
import tensorflow as tf
import warnings


def get_data_run(name, cq=0):
    """
    :param name: 玩法名称
    :return:
    """
    current_number = get_current_number(name)
    logger.info("【{}】最新一期期号：{}".format(name_path[name]["name"], current_number))
    logger.info("正在获取【{}】数据。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    if cq == 1 and name == "kl8":
        data = spider_cq(name, 1, current_number, "train")
    else:
        data = spider(name, 1, current_number, "train")
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】数据准备就绪，共{}期, 下一步可训练模型...".format(name_path[name]["name"], len(data)))
    else:
        logger.error("数据文件不存在！")

def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    url = "https://datachart.500.com/{}/history/".format(name)
    path = "newinc/history.php?start={}&end={}&limit={}"
    if name == "qxc" or name == "pls":
        path = "inc/history.php?start={}&end={}&limit={}"
    elif name == "kl8":
        url = "https://datachart.500.com/{}/zoushi/".format(name)
        path = "newinc/jbzs_redblue.php?from=&to=&shujcount=0&sort=1&expect=-1"
    return url, path

def get_current_number(name):
    """ 获取最新一期数字
    :return: int
    """
    url, _ = get_url(name)
    if name in ["qxc", "pls"]:
        r = requests.get("{}{}".format(url, "inc/history.php"), verify=False)
    elif name in ["ssq", "dlt"]:
        r = requests.get("{}{}".format(url, "history.shtml"), verify=False)
    elif name in ["kl8"]:
        r = requests.get("{}{}".format(url, "newinc/jbzs_redblue.php"), verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    if name in ["kl8"]:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="to")["value"]
    else:
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num

def spider_cq(name="kl8", start=1, end=999999, mode="train", windows_size=0):
    if name == "kl8" and mode == "train":
        url = "https://data.917500.cn/kl81000_cq_asc.txt"
        r = requests.get(url, headers = {'User-agent': 'chrome'})
        data = []
        lines = sorted(r.text.split('\n'), reverse=True)
        for line in lines:
            if len(line) < 10:
                continue
            item = dict()
            line = line.split(',')
            line = line[0].split(' ')
            # item[u"id"] = line[0]
            strdate = line[1].split('-')
            item[u"日期"] = strdate[0] + strdate[1] + strdate[2]
            item[u"期数"] = line[0]  
            for i in range(1, 21):
                item[u"红球_{}".format(i)] = line[i + 1]
            data.append(item)
        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(name_path[name]["path"], data_cq_file_name), encoding="utf-8",index=False)
        return pd.DataFrame(data)
    elif name == "kl8" and mode == "predict":
        ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_cq_file_name))  
        data = []
        if windows_size > 0:
            ori_data = ori_data[0:windows_size]
        for i in range(len(ori_data)):
            item = dict()
            item[u"期数"] = ori_data.iloc[i, 1]
            for j in range(20):
                item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
            data.append(item)
        return pd.DataFrame(data)
    else:
        spider(name, start, end, mode)

def spider(name="ssq", start=1, end=999999, mode="train", windows_size=0):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期
    :param mode 模式，train：训练模式，predict：预测模式（训练模式会保持文件）
    :return:
    """
    if mode == "train":
        url, path = get_url(name)
        limit = int(end) - int(start) + 1
        url = "{}{}".format(url, path.format(int(start), int(end), limit))
        r = requests.get(url=url, verify=False)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        if name in ["ssq", "dlt", "kl8"]:
            trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        elif name in ["qxc", "pls"]:
            trs = soup.find("div", class_="wrap_datachart").find("table", id="tablelist").find_all("tr")
        data = []
        for tr in trs:
            item = dict()
            if name == "ssq":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(6):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                item[u"蓝球"] = tr.find_all("td")[7].get_text().strip()
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(5):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                for j in range(2):
                    item[u"蓝球_{}".format(j+1)] = tr.find_all("td")[6+j].get_text().strip()
                data.append(item)
            elif name == "pls":
                if tr.find_all("td")[0].get_text().strip() == "注数" or tr.find_all("td")[1].get_text().strip() == "中奖号码":
                    continue
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                numlist = tr.find_all("td")[1].get_text().strip().split(" ")
                for i in range(3):
                    item[u"红球_{}".format(i+1)] = numlist[i]
                data.append(item)
            elif name == "kl8":
                tds = tr.find_all("td")
                index = 1
                for td in tds:
                    if td.has_attr('align') and td['align'] == 'center':
                        item[u"期数"] = td.get_text().strip()
                    elif td.has_attr('class') and td['class'][0] == 'chartBall01':
                        item[u"红球_{}".format(index)] = td.get_text().strip()
                        index += 1
                if item:
                    data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")

        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(name_path[name]["path"], data_file_name), encoding="utf-8")
        return pd.DataFrame(data)

    elif mode == "predict":
        ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))  
        data = []
        if windows_size > 0:
            ori_data = ori_data[0:windows_size]
        for i in range(len(ori_data)):
            item = dict()
            if (ori_data.iloc[i, 1] < int(start) or ori_data.iloc[i, 1] > int(end)) and windows_size == 0:
                continue
            if name == "ssq":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(6):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                item[u"蓝球"] = ori_data.iloc[i, 8]
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(5):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                for k in range(2):
                    item[u"蓝球_{}".format(k+1)] = ori_data.iloc[i, 7+k]
                data.append(item)
            elif name == "pls":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(3):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                data.append(item)
            elif name == "kl8":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(20):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")
        return pd.DataFrame(data)

filedata = []
filetitle = []

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

warnings.filterwarnings('ignore')

red_graph = tf.compat.v1.Graph()
blue_graph = tf.compat.v1.Graph()
pred_key_d = {}
red_sess = tf.compat.v1.Session(graph=red_graph)
blue_sess = tf.compat.v1.Session(graph=blue_graph)
mini_args = {}
# current_number = get_current_number(mini_args.name)

def setMiniargs(args):
    global mini_args
    mini_args = args

def init():
    global mini_args,pred_key_d, red_graph, blue_graph, red_sess, blue_sess, filedata, filetitle
    filedata = []
    filetitle = []
    red_graph = tf.compat.v1.Graph()
    blue_graph = tf.compat.v1.Graph()
    pred_key_d = {}
    red_sess = tf.compat.v1.Session(graph=red_graph)
    blue_sess = tf.compat.v1.Session(graph=blue_graph)
    mini_args = {}

def run_predict(window_size):
    global pred_key_d, red_graph, blue_graph, red_sess, blue_sess
    if window_size != 0:
        model_args[mini_args.name]["model_args"]["windows_size"] = window_size
    redpath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]) + model_args[mini_args.name]["subpath"]['red']
    bluepath = model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]) + model_args[mini_args.name]["subpath"]['blue']
    if mini_args.name == "ssq":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(redpath)
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(redpath))
        logger.info("已加载红球模型！窗口大小:{}".format(model_args[mini_args.name]["model_args"]["windows_size"]))
    
        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(bluepath)
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(bluepath))
        logger.info("已加载蓝球模型！窗口大小:{}".format(model_args[mini_args.name]["model_args"]["windows_size"]))

        # 加载关键节点名
        with open("{}/{}".format(model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]), pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(mini_args.name)
        logger.info("【{}】最近一期:{}".format(name_path[mini_args.name]["name"], current_number))

    elif mini_args.name == "dlt":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(redpath)
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(redpath))
        logger.info("已加载红球模型！窗口大小:{}".format(model_args[mini_args.name]["model_args"]["windows_size"]))

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(bluepath)
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(bluepath))
        logger.info("已加载蓝球模型！窗口大小:{}".format(model_args[mini_args.name]["model_args"]["windows_size"]))

        # 加载关键节点名
        with open("{}/{}".format(model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]), pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(mini_args.name)
        logger.info("【{}】最近一期:{}".format(name_path[mini_args.name]["name"], current_number))

    elif mini_args.name in ["pls", "kl8"]:
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(redpath)
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(redpath))
        logger.info("已加载红球模型！窗口大小:{}".format(model_args[mini_args.name]["model_args"]["windows_size"]))

        # 加载关键节点名
        with open("{}/{}".format(model_path + model_args[mini_args.name]["pathname"]['name'] + str(model_args[mini_args.name]["model_args"]["windows_size"]), pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(mini_args.name)
        logger.info("【{}】最近一期:{}".format(name_path[mini_args.name]["name"], current_number))

def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(name, predict_features, windows_size):
    """ 处理异常
    """
    if len(predict_features) != windows_size:
        logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(predict_features) != windows_size:
            # predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
            if mini_args.cq == 0:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict", windows_size)
            else:
                predict_features = spider_cq(name, last_current_year + max_times, get_current_number(name), "predict", windows_size)
            # time.sleep(np.random.random(1).tolist()[0])
            max_times -= 1
        return predict_features
    return predict_features


def get_red_ball_predict_result(predict_features, sequence_len, windows_size):
    """ 获取红球预测结果
    """
    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    if mini_args.name not in ["pls"]:
        hotfixed = 1
    else:
        hotfixed = 0
    data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - hotfixed
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[0][0]])
        pred = red_sess.run(reverse_sequence, feed_dict={
            "inputs:0": data.reshape(1, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    return pred, name_list


def get_blue_ball_predict_result(name, predict_features, sequence_len, windows_size):
    """ 获取蓝球预测结果
    """
    if name == "ssq":
        data = predict_features[[ball_name[1][0]]].values.astype(int) - 1
        with blue_graph.as_default():
            softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(softmax, feed_dict={
                "inputs:0": data.reshape(1, windows_size)
            })
        return pred
    else:
        name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
        data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
        with blue_graph.as_default():
            reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(reverse_sequence, feed_dict={
                "inputs:0": data.reshape(1, windows_size, sequence_len),
                "sequence_length:0": np.array([sequence_len] * 1)
            })
        return pred, name_list


def get_final_result(name, predict_features, mode=0):
    """" 最终预测函数
    """
    m_args = model_args[name]["model_args"]
    if name == "ssq":
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["sequence_len"], m_args["windows_size"])
        blue_pred = get_blue_ball_predict_result(name, predict_features, 0, m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
        pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }
    elif name == "dlt":
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["red_sequence_len"], m_args["windows_size"])
        blue_pred, blue_name_list = get_blue_ball_predict_result(name, predict_features, m_args["blue_sequence_len"], m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
        pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }
    elif name == "pls":
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["red_sequence_len"], m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
        pred_result_list = red_pred[0].tolist()
        return {
            b_name: int(res) for b_name, res in zip(ball_name_list, pred_result_list)
        }
    elif name == "kl8":
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["red_sequence_len"], m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list]
        pred_result_list = red_pred[0].tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }

def predict_run(name):
    global filedata, filetitle
    windows_size = model_args[name]["model_args"]["windows_size"]
    diff_number = windows_size - 1
    current_number = get_current_number(mini_args.name)
    if mini_args.cq == 0:
        data = spider(name, str(int(current_number) - diff_number), current_number, "predict", windows_size)
    else:
        data = spider_cq(name, str(int(current_number) - diff_number), current_number, "predict", windows_size)
    if data is None or len(data) <= 0:
        logger.info("【{}】预测期号：{} 窗口大小:{} 数据为空, 请检查数据文件是否存在，或训练与预测参数是否匹配".format(name_path[name]["name"], int(current_number) + 1, windows_size))
        exit(0)
    logger.info("【{}】预测期号：{} 窗口大小:{}".format(name_path[name]["name"], int(current_number) + 1, windows_size))
    predict_features_ = try_error(name, data, windows_size)
    # logger.info("预测结果：{}".format(get_final_result(name, predict_features_)))
    predict_dict = get_final_result(name, predict_features_)
    ans = ""
    _data = []
    _title = []
    for item in predict_dict:
        if (item == "红球_1" or item == "红球"):
            ans += "红球："
        if (item == "蓝球_1" or item == "蓝球"):
            ans += "蓝球："
        ans += str(predict_dict[item]) + " "
        _data.append(int(predict_dict[item]))
        _title.append(item)
    logger.info("预测结果：{}".format(ans))
    filedata.append(_data.copy())
    filetitle = _title.copy()
    return filedata, filetitle


# if __name__ == "__main__":
#     spider_cq("kl8", "20180101", "20180110", "train")