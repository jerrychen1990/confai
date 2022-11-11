#!/usr/bin/env python
# encoding: utf-8
"""
@author: chenhao
@file: core.py
@time: 2022/11/1 11:37
"""
import copy
import logging
import os
from abc import abstractmethod
from typing import Iterable

from snippets import jdumps, jdump, jload, ensure_dir_path, jload_lines

from confai.models.schema import *
from confai.utils import batch_apply

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    model_name: str
    model_cls: str
    task_config: dict
    tokenizer_config: dict


class ConfAIBaseModel(ABC, object):
    task: Task = None
    """
    所有model的一个基类，可以用底层可以是一个nn model，可以是一个规则系统，也可以是多个其他model构成的pipeline
    一个model用来解决一类给定输入输出的问题
    """

    # 读取配置文件，init一个model实体
    def __init__(self, config: ModelConfig):
        self._load_config(config)
        self.predict_fn_dict = dict()

    def _load_config(self, config: ModelConfig):
        logger.info("loading config...")
        self.config = copy.copy(config)
        logger.info("init model with config:")
        logger.info(jdumps(self.config))
        self.model_name = config.model_name
        self.task_config = config.task_config

    @classmethod
    def read_examples(cls, path: PathOrPaths) -> Iterable[Example]:
        dicts = jload_lines(path, return_generator=True)
        return (cls.task.input_cls(**d) for d in dicts)

    @staticmethod
    def get_config_path(path):
        return os.path.join(path, "config.json")

    @staticmethod
    def _get_assets_path(path):
        return os.path.join(path, "assets")

    @abstractmethod
    def save_assets(self, path):
        raise NotImplementedError

    # save model到path路径下，方便下次可以从path路径复原出model
    @ensure_dir_path
    def save(self, path):
        logger.info(f"saving model to {path}")
        jdump(self.config, ConfAIBaseModel.get_config_path(path))
        self.save_assets(path=self._get_assets_path(path))
        logger.info("save model done")

    # 从$path路径下load出模型
    @classmethod
    def load(cls, path: str = None, config: ModelConfig = None):
        if not config:
            if path:
                logger.info(f"loading model from path:{path}")
                config = jload(cls.get_config_path(path))
                config = ModelConfig(**config)
            else:
                raise ValueError("neither path or config is given!")
        model = cls(config=config)
        return model


# 可以预测的模型
class PredictableModel(ConfAIBaseModel):
    @abstractmethod
    def do_predict(self, examples: ExampleOrExamples, **kwargs) -> PredictOrPredicts:
        raise NotImplementedError

    # predict on examples or dicts or jsonlines
    def predict(self, data: PathOrDictOrExample, **kwargs) -> PredictOrPredicts:
        if isinstance(data, str):
            examples = list(self.read_examples(path=data))
        elif isinstance(data, dict) or (isinstance(data, list) and isinstance(data[0], dict)):
            examples = batch_apply(self.task.input_cls, data)
        else:
            examples = data
        predict = self.do_predict(examples=examples, **kwargs)
        return predict


class TrainableModel(ConfAIBaseModel):
    @abstractmethod
    def train(self, train_data: PathOrDictOrExample, **kwargs):
        raise NotImplementedError
