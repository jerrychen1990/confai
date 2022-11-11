#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: nn_core.py
@time: 2022/11/1 15:50
"""
import os
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np
from snippets import ensure_dir_path, get_batched_data, LogCostContext, adapt_single
from tqdm import tqdm

from confai.models.core import PredictableModel, TrainableModel, ModelConfig
from confai.models.schema import *
from confai.utils import ldict2dlist

logger = logging.getLogger(__name__)

Feature = Union[str, int, float, np.ndarray]


class AbstractDataManager:
    HOME_ENV = None

    @classmethod
    @abstractmethod
    def get_tokenizer(cls, tokenizer_path: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def collate(cls, features: List[Dict[str, List]], tokenizer, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def get_local_path(cls, path: str):
        if path.startswith("/"):
            return path
        if cls.HOME_ENV in os.environ:
            local_path = os.path.join(os.environ[cls.HOME_ENV], path)
            if os.path.exists(local_path):
                return local_path
            return path
        return path

    @classmethod
    @abstractmethod
    def save_tokenizer(cls, tokenizer, path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_dataset(cls, data_or_path: PathOrDictOrExample, task: Task, map_fn: Callable):
        raise NotImplementedError


class NNModelConfig(ModelConfig):
    tokenizer_config: dict


class NNModel(PredictableModel, TrainableModel):
    data_manager: type(AbstractDataManager) = None
    default_nn_model_type = None
    input_keys = None
    output_keys = None

    def __init__(self, config: NNModelConfig):
        super().__init__(config)
        self.nn_model = None
        self.save_model_fn_dict = dict()
        self.load_model_fn_dict = dict()
        self.predict_fn_dict["model"] = self.predict_with_model
        self.tokenizer = self.data_manager.get_tokenizer(**self.tokenizer_config)

    def _load_config(self, config: NNModelConfig):
        super()._load_config(config=config)
        self.tokenizer_config = config.tokenizer_config

    @ensure_dir_path
    def save(self, path, save_tokenizer=True,
             save_nn_model=True, nn_model_types: List = None):
        super().save(path=path)
        if save_tokenizer:
            tokenizer_path = os.path.join(path, "tokenizer")
            self.save_tokenizer(path=tokenizer_path)
        if save_nn_model:
            nn_model_path = os.path.join(path, "nn_model")
            self.save_nn_model(path=nn_model_path, nn_model_types=[
                self.default_nn_model_type] if nn_model_types is None else nn_model_types)

    @ensure_dir_path
    def save_nn_model(self, path, nn_model_types: List):
        if self.nn_model:
            for nn_model_type in nn_model_types:
                if nn_model_type not in self.save_model_fn_dict:
                    logger.warning(f"saving nn_model_type:{nn_model_type} is not supported by {self.__class__}!")
                else:
                    logger.info(f"saving nn_model with type:{nn_model_type}")
                    save_fn = self.save_model_fn_dict[nn_model_type]
                    save_fn(path=path)
            logger.info("save nn model done")

    @ensure_dir_path
    def save_tokenizer(self, path):
        logger.info(f"saving tokenize to {path}")
        self.data_manager.save_tokenizer(self.tokenizer, path)

    @classmethod
    def load(cls, path: str = None, config: dict = None, load_nn_model=True):
        model = super().load(path=path, config=config)
        if load_nn_model:
            if not path:
                logger.warning("path must be given to load nn_model! pass loading nn model!")
            else:
                nn_model_path = os.path.join(path, "nn_model")
                model.load_nn_model(path=nn_model_path)
        logger.info("load model done")
        return model

    def load_nn_model(self, path, nn_model_type=None):
        if nn_model_type is None:
            nn_model_type = self.default_nn_model_type
        if nn_model_type not in self.load_model_fn_dict:
            logger.warning(f"loading nn_model_type:{nn_model_type} is not supported by {self.__class__}!")
        else:
            load_fn = self.load_model_fn_dict[nn_model_type]
            self.nn_model = load_fn(path=path)

    @abstractmethod
    def example2feature(self, example: Example, mode: str) -> Dict[str, Feature]:
        raise NotImplementedError

    @abstractmethod
    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> PredictOrPredicts:
        raise NotImplementedError

    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    def add_predict_fn(self, mode: str, predict_fn: Callable):
        self.predict_fn_dict[mode] = predict_fn

    @abstractmethod
    def predict_with_model(self, features: Dict[str, np.ndarray], **kwargs) -> Dict[str, Feature]:
        raise NotImplementedError

    @adapt_single(ele_name="examples")
    def do_predict(self, examples: ExampleOrExamples, mode="model", batch_size=32, **kwargs) -> PredictOrPredicts:
        # logger.info(examples)
        if mode not in self.predict_fn_dict:
            raise ValueError(f"invalid predict mode:{mode}!")
        predict_fn = self.predict_fn_dict[mode]
        features = (self.example2feature(e, mode="test") for e in examples)
        dataset = get_batched_data(features, batch_size)
        # dataset = self.load_dataset(data_or_path=examples)
        preds = []
        with LogCostContext(name="predict batches"):
            for features in tqdm(dataset):
                # logger.info(f"{features=}")
                to_pred_features = [{k: f[k] for k in self.input_keys} for f in features]
                to_pred_features = self.data_manager.collate(features=to_pred_features, tokenizer=self.tokenizer,
                                                             padding=True, return_tensors="np")
                # logger.debug(f"{to_pred_features=}")
                pred_features = predict_fn(features=to_pred_features, **kwargs)
                pred_features = ldict2dlist(pred_features)
                # logger.debug(f"{pred_features=}")
                assert len(features) == len(pred_features)
                batch_preds = [self.feature2predict(f, pf) for f, pf in zip(features, pred_features)]
                preds.extend(batch_preds)
        return preds

    def train(self, train_data: PathOrDictOrExample,
              eval_data: PathOrDictOrExample = None,
              train_kwargs=dict(),
              callback_kwargs=dict()):
        raise NotImplementedError
