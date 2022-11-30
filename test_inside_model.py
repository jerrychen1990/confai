#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_inside_model.py
@time: 2022/11/16 10:27
"""
import os

os.environ["CONFAI_DEBUG"] = "True"
CONFAI_PATH = "."
os.environ["EXPERIMENT_HOME"] = "/Users/chenhao/experiment"


from confai.models import load_model
from confai.experiments import get_model_config, ExperimentConfig
from confai.utils import *

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


model_kwargs = exp_config.nn_model_config
nn_model = model.build_model(**model_kwargs)


import copy
import torch

train_kwargs = copy.copy(exp_config.train_config)
train_kwargs.update(output_dir="./output", batch_size=2)
print(train_kwargs)
callback_kwargs = copy.copy(exp_config.callback_config)
if 'eval_save' in callback_kwargs:
    kwargs = callback_kwargs["eval_save"]
    kwargs.update(model=model, eval_data=eval_data_path, test_config=exp_config.test_config,experiment_path=".")

model.train(train_data=train_data_path, eval_data=eval_data_path, train_kwargs=train_kwargs, callback_kwargs=callback_kwargs)