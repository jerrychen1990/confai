#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: common.py
@time: 2022/11/8 17:37
"""
import logging
from abc import ABC

from confai.models.core import ConfAIBaseModel, ModelConfig
from confai.models.schema import Task

logger = logging.getLogger(__name__)


class BaseTextGenModel(ConfAIBaseModel, ABC):
    task = Task.TEXT_GEN

    def _load_config(self, config: ModelConfig):
        super()._load_config(config)
        self.max_len = self.task_config["max_len"]
