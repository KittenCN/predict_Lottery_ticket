# 彩票AI预测（目前支持双色球，大乐透，排列三，快乐8, 七星彩，福彩3D）

##  -------------------------------------
##  一定要用get_data下载数据，并用train一定次数后，才能使用predict预测，默认没有数据和模型
##  如果train发生错误，请优先检查主目录下是否有data, model, predict三个空目录，没有的话自行建立下，我好像忘记让它自动生成了，也懒得写了
##  -------------------------------------
##  现有的模型对于七星彩，福彩3D，排列三等允许重复数字的彩票，效果可能有不好，因为我只是简单的把数字当成了类别，没有更多的考虑到数字之间的关系，而且由于时间原因，我也没有详细测试，所以，如果想要更加准确的预测这些彩票，可以自己修改代码，或者将建议留下，等我有空了再更新
##  -------------------------------------

项目思路来自自：https://github.com/zepen/predict_Lottery_ticket
已将80%的代码重写，并按照我自己的思路进行了强化和修改。
自动选择并同时支持CPU和GPU计算。CPU使用原本的Keras LSTM，GPU使用CudnnLSTM，同等参数下，GPU效率高于CPU，时间窗口越大，batch_size越大，效率差就越明显；建议有好显卡的朋友使用GPU训练.
目前我正在修改网络结构，并迁移到我比较熟悉的pytorch框架之下: https://github.com/KittenCN/predict_Lottery_ticket_pytorch

## New
* 20230903
* 1. 增加了两个kl8_开头的文件，用来测试计算一些数据预处理的公式，以及计算预测号码获奖情况, 正常情况无需使用
* 20230327
* 1. 增加对于七星彩（qxc），福彩3D（sd）的支持
* 20230322:
* 1. 增加执行参数开关，针对red_epochs，blue_epochs，batch_size三个参数，如果值为-1，则读取config文件的值，且修改三个参数的默认值为-1

## Installing
        
* step1，安装anaconda(可参考https://zhuanlan.zhihu.com/p/32925500)；

* step2，创建一个conda环境，conda create -n your_env_name python=3.8；
       
* step3，进入创建conda的环境 conda activate your_env_name，然后根据自己机器的状况，选择CPU或者GPU模式，并在requirement文件中，把对应版本的Tensorflow解除注释，并执行pip install -r requirements.txt；如果不确定哪个版本更合适，建议使用gpu版本
* 备注：根据我个人的测试，不推荐使用其他版本的tensorflow,如果因为硬件原因，一定要用更高或者更低版本的tensorflow,请同时更新tensorflow-addons，pandas，numpy的版本。
       
* step4，按照Getting Started执行即可

## Getting Started

```python
python get_data.py  --name ssq  # 执行获取双色球训练数据
```
如果出现解析错误，应该看看网页 http://datachart.500.com/ssq/history/newinc/history.php 是否可以正常访问
若要大乐透，替换参数 --name dlt 即可

```python
python run_train_model.py --name ssq  --windows_size 3,5,7 --red_epochs 1 --blue_epochs 1 --batch_size 1  # 执行训练双色球模型
``` 
开始模型训练，先训练红球模型，再训练蓝球模型，模型参数和超参数在 config.py 文件中自行配置
具体训练时间消耗与模型参数和超参数相关。
若要多个窗口尺寸依次训练，替换参数 --windows_size 3,5,7 即可
red_epochs 为红球训练次数
blue_epochs 为篮球训练次数
batch_size 为每轮训练的数量

```python
python run_predict.py  --name ssq --windows_size 3,5,7  # 执行双色球模型预测
```
预测结果会打印在控制台

# 注意事项：
1. 使用高于1个batch_size训练后，不能立即预测，必须使用1个batch_size再次训练保存才可以，应该是batch_size维度被保存在inputs里面的原因，也可使用--predict_pro 1 参数进行这个动作
2. 使用GPU推导时使用的是RNN的CudnnLSTM而非Keras的LSTM，因此两个模型保存的checkpoint不通用！

