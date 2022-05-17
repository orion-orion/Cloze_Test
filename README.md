<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-02-05 18:23:00
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-05-16 19:19:45
-->
### 关于本项目
本项目为2020-2021年ASC世界大学生超级计算机竞赛第3题，题目为训练机器学习模型完成一个完形填空形式的NLP任务，我们采用ALBert模型，使用赛方给定的数据集进行微调和测试, 最终在ALBert-xxlarge的预训练模型下达到90%的准确率


### 环境配置
在我的机子上是Python3.6和Pytorch1.2


### 数据集
数据集直接采用的赛方给定的完形填空题目数据集，放在项目目录中的task3文件夹下

### 语料库
ALBert-xxlarge模型的预料库我已经下载，即项目中的model/albert-xxlarge-uncased-vocab.txt文件

### 预训练模型和我自己训练的最终模型下载
预训练模型和我自己最终训练的模型我已经上传到Google drive，链接如下：  
链接: https://pan.baidu.com/s/1eukkVlRrYj72_GGJTQiXMQ  


### 数据预处理
先运行:
```
python3 data_util.py
```
（注意：若有需要，你需要在这里修改data_util.py中语料库的本地路径）


### 模型微调/预测 
运行:
```
./run.sh
```
（同样，若有需要，你在模型微调你可以在run.sh中修改预训练模型的本地路径，预测时你需要修改本地最终模型的路径）

