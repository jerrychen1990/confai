#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: inside_model.py
@time: 2022/11/28 16:53
"""
import sys
import os

os.environ["CONFAI_DEBUG"] = "True"
CONFAI_PATH = os.environ["CONFAI_PATH"]
if CONFAI_PATH not in sys.path:
    sys.path.insert(0, CONFAI_PATH)

import logging

logger = logging.getLogger(__name__)

from confai.models import get_model_cls, load_model
from confai.schema import *
from confai.experiments import get_model_config, ExperimentConfig
from confai.utils import *

config_path = f"{CONFAI_PATH}/conf/examples/mlm_random.ini"

exp_config = read_config(config_path)
exp_config = ExperimentConfig(**exp_config)
logger.info(exp_config.dict())

model_config = get_model_config(exp_config)

model = load_model(config=model_config, load_nn_model=False)
tokenizer = model.tokenizer

train_data_path = exp_config.data_config.train_data_path
train_data = list(model.read_examples(train_data_path))
logger.info(("train_data num:", len(train_data)))

model_kwargs = exp_config.nn_model_config
nn_model = model.build_model(**model_kwargs)

import copy
import torch


train_kwargs = copy.copy(exp_config.train_config)
train_kwargs.update(output_dir="./output")
callback_kwargs = copy.copy(exp_config.callback_config)
model.train(train_data=train_data, eval_data=None, train_kwargs=train_kwargs, callback_kwargs=callback_kwargs)
model.train(train_data=train_data, eval_data=None, train_kwargs=train_kwargs, callback_kwargs=callback_kwargs)
model.train(train_data=train_data, eval_data=None, train_kwargs=train_kwargs, callback_kwargs=callback_kwargs)
