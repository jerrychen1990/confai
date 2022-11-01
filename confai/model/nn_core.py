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
from typing import Dict, Callable

import numpy as np
from snippets import ensure_dir_path, get_batched_data, LogCostContext
from tqdm import tqdm

from confai.model.core import PredictableModel, TrainableModel
from confai.model.schema import *
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


class NNModel(PredictableModel, TrainableModel):
    data_manager: type(AbstractDataManager) = None
    model_cls = None
    input_keys = None
    output_keys = None

    def __init__(self, config):
        super().__init__(config)
        self.nn_model = None
        self.save_model_fn_dict = dict()
        self.load_model_fn_dict = dict()
        self.predict_fn_dict["model"] = self.predict_with_model
        self.tokenizer = self.data_manager.get_tokenizer(**self.config["tokenizer_config"])

    def _load_config(self, config):
        super()._load_config(config=config)
        self.config["model_cls"] = self.model_cls
        if self.config.get("input_keys") and not self.input_keys:
            self.input_keys = self.config["input_keys"]
        if self.config.get("output_keys") and not self.output_keys:
            self.output_keys = self.config["output_keys"]
        logger.debug(f"{self.input_keys=}")
        logger.debug(f"{self.output_keys=}")
        self.config["input_keys"] = self.input_keys
        self.config["output_keys"] = self.output_keys

    @ensure_dir_path
    def save(self, path, save_type="json", save_tokenizer=True, **kwargs):
        super().save(path=path, save_type=save_type)
        if save_tokenizer:
            tokenizer_path = os.path.join(path, "tokenizer")
            self.save_tokenizer(path=tokenizer_path)

    @ensure_dir_path
    def save_tokenizer(self, path):
        logger.info(f"saving tokenize to {path}")
        self.data_manager.save_tokenizer(self.tokenizer, path)

    @abstractmethod
    def example2features(self, example: Example) -> Dict[str, any]:
        raise not ImportError

    def load_dataset(self, data_or_path: PathOrDictOrExample):
        return self.data_manager.load_dataset(data_or_path=data_or_path, task=self.task,
                                              map_fn=self.example2features)

    @abstractmethod
    def features2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> Predict:
        raise not ImportError

    @ensure_dir_path
    def save(self, path, save_type="json", save_tokenizer=True, save_nn_model=True, nn_model_types=["onnx"],
             **kwargs):
        super().save(path=path, save_type=save_type, save_tokenizer=save_tokenizer)
        if save_nn_model:
            self.save_nn_model(path=path, nn_model_types=nn_model_types)

    def save_nn_model(self, path, nn_model_types: List):
        if self.nn_model:
            nn_model_path = os.path.join(path, "nn_model")
            for nn_model_type in nn_model_types:
                if nn_model_type not in self.save_model_fn_dict:
                    logger.warning(f"saving nn_model_type:{nn_model_type} is not supported by {self.__class__}!")
                else:
                    logger.info(f"saving nn_model with type:{nn_model_type}")
                    save_fn = self.save_model_fn_dict[nn_model_type]
                    save_fn(path=nn_model_path)
            logger.info("save nn model done")

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
    def build_model(self, **kwargs):
        raise NotImplementedError

    def example2train_features(self, example: Example) -> Dict[str, Feature]:
        assert example.get_ground_truth() is not None
        features = self.example2features(example=example)
        self._update_train_features(example=example, features=features)
        return features

    @abstractmethod
    def _update_train_features(self, example, features: Dict[str, Feature]) -> Dict[str, Feature]:
        raise NotImplementedError

    def load_dataset(self, data_or_path, mode="train"):
        assert mode in ["infer", "train"]
        # logger.debug(f"{mode=}")
        if mode == "infer":
            return super().load_dataset(data_or_path=data_or_path)

        return self.data_manager.load_dataset(data_or_path=data_or_path, task=self.task,
                                              map_fn=self.example2train_features)

    @classmethod
    def _get_onnx_model_path(cls, path):
        return os.path.join(path, "model.onnx")

    @abstractmethod
    def predict_with_model(self, features: Dict[str, np.ndarray], **kwargs) -> Dict[str, Feature]:
        raise NotImplementedError

    def add_predict_fn(self, mode: str, predict_fn: Callable):
        self.predict_fn_dict[mode] = predict_fn

    def do_predict(self, examples: List[Example], mode="model", batch_size=32, **kwargs) -> List[Predict]:
        if mode not in self.predict_fn_dict:
            raise ValueError(f"invalid predict mode:{mode}!")
        predict_fn = self.predict_fn_dict[mode]
        features = (self.example2features(e) for e in examples)
        dataset = get_batched_data(features, batch_size)
        # dataset = self.load_dataset(data_or_path=examples)
        preds = []
        with LogCostContext(name="predict batches"):
            for features in tqdm(dataset):
                logger.debug(f"{features=}")
                to_pred_features = [{k: f[k] for k in self.config["input_keys"]} for f in features]
                to_pred_features = self.data_manager.collate(features=to_pred_features, tokenizer=self.tokenizer,
                                                             padding=True, return_tensors="np")
                # logger.debug(f"{to_pred_features=}")
                pred_features = predict_fn(features=to_pred_features, **kwargs)
                pred_features = ldict2dlist(pred_features)
                # logger.debug(f"{pred_features=}")
                assert len(features) == len(pred_features)
                batch_preds = [self.features2predict(f, pf) for f, pf in zip(features, pred_features)]
                preds.extend(batch_preds)
        return preds
