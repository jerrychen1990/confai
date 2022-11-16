#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_inside_model.py
@time: 2022/11/16 10:27
"""
import sys
import os

from transformers.tokenization_utils_base import TruncationStrategy

os.environ["CONFAI_DEBUG"] = "True"
CONFAI_PATH = "."
os.environ["EXPERIMENT_HOME"] = "/Users/chenhao/experiment"


from confai.models import get_model_cls, load_model
from confai.models.schema import *
from confai.experiments import get_model_config, ExperimentConfig
from confai.utils import *
import dataclasses

config_path = f"{CONFAI_PATH}/conf/examples/sentiment_cls_token_cls.ini"
# config_path = f"{CONFAI_PATH}/conf/examples/poem_transform_gen.ini"
# config_path = f"{CONFAI_PATH}/conf/text_gen/poem_transform_gen_sample.ini"
config_path = f"{CONFAI_PATH}/conf/examples/ner_seqlabel.ini"
config_path = f"{CONFAI_PATH}/conf/examples/sentiment_cls_mlm_cls.ini"

exp_config = read_config(config_path)
exp_config = ExperimentConfig(**exp_config)
exp_config.dict()

model_config = get_model_config(exp_config)
model_config

model = load_model(config=model_config, load_nn_model=False)
model
tokenizer = model.tokenizer


# feature = tokenizer(text="很棒很棒很棒很棒很棒", text_pair="[MASK]", text_target="很棒", text_pair_target="好",
#                          truncation=TruncationStrategy.ONLY_FIRST,
#                          max_length=10)
#
# print(feature)

# tokenizer
# model.prompts
# model.token2label
#
train_data_path = exp_config.data_config.train_data_path
train_data = list(model.read_examples(train_data_path))
"train_data num:", len(train_data)

eval_data_path = exp_config.data_config.eval_data_path
eval_data = list(model.read_examples(eval_data_path))
"eval_data num:", len(eval_data)

test_data_path = exp_config.data_config.test_data_path
test_data = list(model.read_examples(test_data_path))
"test_data num:", len(test_data)

print(train_data[0])
#
train_dataset = model.load_dataset(train_data_path)
print(train_dataset[0])
