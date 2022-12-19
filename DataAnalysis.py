import pandas as pd
from config import *

datacnt = [0] * 81
dataori = [i for i in range(81)]
ori_data = []

def BasicAnalysis(oridata):
    # Basic analysis of the data
    # ori_data: original data
    # Return: None
    # Author: KittenCN
    global datacnt, dataori
    for row in oridata:
        for item in row:
            datacnt[item] += 1
    datacnt, dataori = sortcnt(datacnt, dataori, 81)
    lastcnt = -1
    for i in range(81):
        if dataori[i] == 0:
            continue
        if lastcnt != datacnt[i]:
            print()
            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
            lastcnt = datacnt[i]
        elif lastcnt == datacnt[i]:
            print(dataori[i], end = " ")
    return datacnt, dataori
    
def sortcnt(datacnt, dataori, rangenum=81):
    for i in range(rangenum):
        for j in range(i + 1, rangenum):
            if datacnt[i] < datacnt[j]:
                datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                dataori[i], dataori[j] = dataori[j], dataori[i]
            elif datacnt[i] == datacnt[j]:
                if dataori[i] < dataori[j]:
                    datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                    dataori[i], dataori[j] = dataori[j], dataori[i]
    return datacnt, dataori

def getdata():
    strdata = input("Shrinkage ratio (split by ',' ): ").split(',')
    data = [int(i) for i in strdata]
    oridata = []
    for i in range(81):
        if dataori[i] == 0:
            continue
        if datacnt[i] in data:
            oridata.append(dataori[i])
    booldata = [False] * len(oridata)
    return oridata, booldata

def dfs(oridata, booldata, getnums, dep, ans, cur):
    if dep == getnums:
        ans.append(cur.copy())
        return
    for i in oridata:
        if booldata[oridata.index(i)] or i <= cur[dep - 1]:
            continue
        booldata[oridata.index(i)] = True
        cur[dep] = i
        dfs(oridata,booldata, getnums, dep + 1, ans, cur)
        booldata[oridata.index(i)] = False
    return ans

def shrink(oridata, booldata):
    getnums = int(input("How many numbers do you want to get? (-1 means over) "))
    while getnums != -1:
        ans = dfs(oridata,booldata, getnums, 0, [], [0] * getnums)
        print("There are {} ways to get {} numbers.".format(len(ans), getnums))
        strSumMinMax = input("input the min and max of the numbers? (split by ',') ").split(',')
        SumMinMax = [int(i) for i in strSumMinMax]
        SumMin = SumMinMax[0]
        SumMax = SumMinMax[1]
        for i in range(len(ans)):
            if sum(ans[i]) < SumMin or sum(ans[i]) > SumMax:
                continue
            print(ans[i])
        getnums = int(input("How many numbers do you want to get? (-1 means over) "))

def hisanalysis():
    oridata = pd.read_csv("{}{}".format(name_path["kl8"]["path"], data_file_name))
    data = oridata.iloc[:, 2:].values
    sumori = [i for i in range(1401)]
    sumcnt = [0] * 1401
    for row in data:
        sum = 0
        for item in row:
            sum += item
        sumcnt[sum] += 1
    sumcnt, sumori = sortcnt(sumcnt, sumori, 1401)
    sumtop = int(input("input num of the top sumcnt:"))
    lastsum = -1
    sumans = []
    sumanscnt = 0
    for i in range(1401):
        if sumcnt[i] == 0:
            continue
        if sumcnt[i] != lastsum:
            if sumanscnt == sumtop:
                break;
            else:
                lastsum = sumcnt[i]
                sumanscnt += 1
        sumans.append(sumori[i])
    print(sumans)

if __name__ == "__main__":
    while True:
        choice = int(input("input your choice:"))
        if choice == 1:
            _datainrow = []
            n = int(input("how many data do you want to generate? -1 means get data from file:"))
            if n != -1:
                for i in range(n):
                    tmpdata = input("please input the data #{}: ".format(i + 1)).strip().split(' ')
                    for item in tmpdata:
                        _datainrow.append(int(item))
                    _datainrow.append(int(item) for item in tmpdata)
                    ori_data.append(tmpdata)
            else:
                filename = input("input the file name: ")
                fileadd = "{}{}{}{}".format(predict_path, "kl8/", filename, ".csv")
                ori_data = pd.read_csv(fileadd).values
                for row in ori_data:
                    for item in row:
                        _datainrow.append(item)           
            datacnt, dataori = BasicAnalysis(ori_data)
            print()
            currentnums = input("input current numbaers, -1 means oever: ").split(' ')
            if currentnums[0] != -1:
                curnums = [int(i) for i in currentnums]
                curcnt = 0
                for item in curnums:
                    for row in ori_data:
                        if item in row:
                            curcnt += 1
                            break
                totalnums = len(list(set(_datainrow)))
                print("total success / total nums: {} / {}".format(curcnt, totalnums))
                lastcnt = -1
                for i in range(81):
                    if dataori[i] == 0:
                        continue
                    elif dataori[i] in curnums:
                        if lastcnt != datacnt[i]:
                            print()
                            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
                            lastcnt = datacnt[i]
                        elif lastcnt == datacnt[i]:
                            print(dataori[i], end = " ")
                print()

        elif choice == 2:
            oridata, booldata = getdata()
            print()
            shrink(oridata, booldata)
            print()
        elif choice == 3:
            hisanalysis()
        if choice == 0:
            break
