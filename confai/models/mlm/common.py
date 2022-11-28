#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: common.py
@time: 2022/11/3 11:20
"""
import logging
from abc import ABC

from confai.models.core import ConfAIBaseModel
from confai.schema import Task

logger = logging.getLogger(__name__)


class BaseMLMModel(ConfAIBaseModel, ABC):
    task = Task.MLM

    def _load_config(self, config):
        super()._load_config(config)
        self.max_len = self.task_config["max_len"]

    def save_assets(self, path):
        pass
