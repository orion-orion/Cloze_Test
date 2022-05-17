'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-05 18:23:00
LastEditors: ZhangHongYu
LastEditTime: 2022-05-17 16:26:12
'''
import os
import sys
import json
import argparse
from transformers import AlbertTokenizer
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
file_path = sys.argv[1]
#bert_model = BertForMaskedLM.from_pretrained('/data/jianghao/ralbert-cloth/model/albert-xxlarge-v2/pytorch_model.bin')
PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
bert_tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
max=-1
cnt=0
tot=0
for file in os.listdir(file_path):
    if file.endswith(".json"):
        with open(os.path.join(file_path,file),'r') as f:
            dict = json.load(f)
            sentences=dict['article'].split('.')
            str=""
            for sentence in sentences:
                sentence=sentence.replace('_','[MASK]')
                tokens = bert_tokenizer.tokenize(sentence)
                if len(tokens) == 0:
                    continue
                if tokens[0] != CLS:
                    tokens = [CLS] + tokens
                if tokens[-1] != SEP:
                    tokens.append(SEP)
            str = ''.join(tokens)
            # print(str)
            # print('完了')
            tot=tot+1
            if len(str)>max: 
                max=len(str)
            if len(str)>512:
                cnt=cnt+1
               #os.system("rm "+os.path.join(file_path,file))
print(cnt/tot)

