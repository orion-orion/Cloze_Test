<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-02-05 18:23:00
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-05-17 22:03:13
-->

# 2020-2021年ASC第3题: 使用ALBert模型完成完形填空的NLP任务

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/Cloze_Test)
[![](https://img.shields.io/github/license/orion-orion/Cloze_Test)](https://github.com/orion-orion/Cloze_Test/blob/master/LICENSE)
[![](https://img.shields.io/github/stars/orion-orion/Cloze_Test?style=social)](https://github.com/orion-orion/Cloze_Test)
[![](https://img.shields.io/github/issues/orion-orion/Cloze_Test)](https://github.com/orion-orion/Cloze_Test/issues)
### 关于本项目
本项目为2020-2021年ASC世界大学生超级计算机竞赛第3题，题目为训练机器学习模型完成一个完形填空形式的NLP任务，我们采用ALBert模型，使用赛方给定的数据集进行微调和测试, 最终在ALBert-xxlarge的预训练模型下达到89%的准确率。


### 环境依赖
运行以下命令安装环境依赖：

```shell
pip install -r requirements.txt
```


### 数据集
数据集直接采用的赛方给定的完形填空题目数据集，放在项目目录中的task3文件夹下

### 语料库
ALBert-xxlarge模型的预料库我已经下载，即项目中的model/albert-xxlarge-uncased-vocab.txt文件

### 预训练模型和我自己微调后的模型下载
我已将预训练模型和微调后的模型上传到Google drive，链接如下：  
[Google drive 下载链接](https://drive.google.com/drive/folders/1a1yQemukD8-m-XcOXg3akyxE_6tVgO1L?usp=sharing)


### 数据预处理
先运行:
```
python3 data_util.py
```
（注意：若有需要，你可以修改`data_util.py`中语料库的本地路径）


### 模型预测 
根据赛方数据预测出答案，运行（需要需要先训练好微调后的模型，上面已给出下载链接）:
```
python -u main.py \
    --output_dir debug-exp/ \
    --do_predict \
    --bert_model albert-xxlarge-v2
```
（同样，若有需要，你可以`main.py`中修改预训练模型的本地路径，或在`main.py`中修改微调后模型的保存路径）

