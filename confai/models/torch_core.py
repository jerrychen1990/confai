#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: torch_core.py
@time: 2022/11/1 17:55
"""

import logging
import os
import os.path
from abc import abstractmethod
from typing import Dict, Callable

import numpy as np
import torch
from snippets import ensure_dir_path, log_cost_time, LogCostContext
from transformers import BertConfig, EarlyStoppingCallback, \
    TrainingArguments, Trainer, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.bert import BertOnnxConfig
from transformers.utils import PaddingStrategy
from datasets import Dataset, DatasetDict

from confai.models.callbacks import EvalSaveCallback, get_callbacks
from confai.models.nn_core import NNModel, Feature, AbstractDataManager
from confai.models.schema import *
from confai.utils import safe_call, safe_build_data_cls, dlist2ldict

logger = logging.getLogger(__name__)

_train_arg_map = {
    "batch_size": "per_device_train_batch_size",
    "eval_batch_size": "per_device_eval_batch_size",
    "epochs": "num_train_epochs",
}

_onnx_config_map = {
    BertConfig: BertOnnxConfig
}

_task_map = {
    Task.TEXT_CLS: "sequence-classification",
    Task.TEXT_SPAN_CLS: "token-classification",
}


def get_train_args(**kwargs):
    for s, t in _train_arg_map.items():
        if s in kwargs and t not in kwargs:
            kwargs[t] = kwargs[s]
    logging_pct = kwargs.get("logging_pct")
    if logging_pct:
        if "max_steps" in kwargs:
            max_steps = kwargs["max_steps"]
        else:
            max_steps = kwargs["num_train_epochs"] * kwargs["train_num"] // kwargs["per_device_train_batch_size"]
        logging_steps = int(max_steps * logging_pct)
        kwargs["logging_steps"] = logging_steps
    logger.info(f"train kwargs:{kwargs}")

    training_args = safe_build_data_cls(
        TrainingArguments, kwargs
    )
    return training_args


class HFDataManager(AbstractDataManager):
    HOME_ENV = "HUGGING_FACE_MODEL_HOME"
    _input_keys = ['input_ids', 'token_type_ids', 'attention_mask']

    @classmethod
    def get_tokenizer(cls, tokenizer_path: str):
        local_path = cls.get_local_path(tokenizer_path)
        logger.info(f"initializing tokenizer with path:{local_path}...")
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        return tokenizer

    @classmethod
    def save_tokenizer(cls, tokenizer, path):
        tokenizer.save_pretrained(path)

    @classmethod
    def collate(cls, features: List[Dict[str, List]], tokenizer: PreTrainedTokenizerBase,
                padding: Union[bool, str, PaddingStrategy] = True,
                max_length: Optional[int] = None,
                pad_to_multiple_of: Optional[int] = None,
                label_pad_token_id: int = 0,
                return_tensors: str = "pt") -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        features = [{k: f[k] for k in cls._input_keys if k in f} for f in features]
        # logger.debug(f"{features=}")
        batch = tokenizer.pad(
            features,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        return batch

    @classmethod
    def load_dataset(cls, data_or_path: PathOrDictOrExample, task: Task, map_fn: Callable):
        # read from file or files
        if isinstance(data_or_path, str) or (isinstance(data_or_path, List) and isinstance(data_or_path[0], str)):
            dataset = Dataset.from_json(data_or_path)
        else:
            # read from example or examples
            if isinstance(data_or_path, Example) or \
                    (isinstance(data_or_path, List) and isinstance(data_or_path[0], Example)):
                data_or_path = data_or_path if isinstance(data_or_path, List) else [data_or_path]
                data = [e.dict(exclude={"extra"}) for e in data_or_path]
            # read from dict or list of dicts
            else:
                data = data_or_path if isinstance(data_or_path, List) else [data_or_path]
            # logger.info(data)
            data = dlist2ldict(data)
            logger.debug(data)
            dataset = Dataset.from_dict(mapping=data)

        def transfer2features(item):
            # logger.info(item)
            example = task.input_cls(**item)
            features = map_fn(example)
            return features

        return dataset.map(transfer2features)


class HFTorchModel(NNModel, ABC):
    data_manager = HFDataManager
    default_nn_model_type = "torch"
    nn_model_call = "forward"

    def __init__(self, config):
        super().__init__(config)
        self.save_model_fn_dict["torch"] = self.save_torch_model
        self.load_model_fn_dict["torch"] = self.load_torch_model
        # self.save_model_fn_dict["onnx"] = self.save_onnx_model

    @classmethod
    def _get_torch_model_path(cls, path):
        return os.path.join(path, "model.torch")

    @ensure_dir_path
    def save_torch_model(self, path):
        torch_path = self._get_torch_model_path(path=path)
        logger.info(f"saving torch model to path:{torch_path}")
        torch.save(self.nn_model, torch_path)
        logger.info("saving torch model done")

    def load_torch_model(self, path):
        torch_path = self._get_torch_model_path(path=path)
        logger.info(f"loading nn_model from {torch_path}")
        nn_model = torch.load(torch_path)
        logger.info("load torch model done")
        return nn_model

    def load_dataset(self, data_or_path, mode="train"):
        assert mode in ["infer", "train"]

        return self.data_manager.load_dataset(data_or_path=data_or_path, task=self.task,
                                              map_fn=lambda e: self.example2feature(example=e, mode=mode))

    def predict_with_model(self, features: Dict[str, np.ndarray], **kwargs) -> Dict[str, Feature]:
        self.nn_model.eval()
        with torch.no_grad():
            tensor_features = {k: torch.from_numpy(v).to(self.nn_model.device) for k, v in features.items()}
            logger.debug(tensor_features)
            # output = nn_model(**tensor_features, return_dict=True)
            if self.nn_model_call == "forward":
                predict_fn = self.nn_model.forward
                output = safe_call(predict_fn, **tensor_features, return_dict=True)

            else:
                predict_fn = self.nn_model.generate
                output = safe_call(predict_fn, inputs=tensor_features["input_ids"],
                                   **kwargs, return_dict_in_generate=True)
            output = {k: output[k].cpu().numpy() for k, v in output.items()}
        return output

    def collator_fn(self, features, **kwargs):
        batch = self.data_manager.collate(features=features, tokenizer=self.tokenizer, **kwargs)
        # logger.debug(f"{batch}")
        self._update_batch(batch=batch, features=features)
        return batch

    @abstractmethod
    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        raise NotImplementedError

    @log_cost_time(name="train phase")
    def train(self, train_data: PathOrDictOrExample,
              eval_data: PathOrDictOrExample = None,
              train_kwargs=dict(),
              callback_kwargs=dict()):
        datasets = dict()
        with LogCostContext(name="load train/eval data"):
            train_dataset = self.load_dataset(train_data, mode="train")
            # logger.info(train_dataset)
            datasets["train_dataset"] = train_dataset
            if eval_data:
                eval_dataset = self.load_dataset(eval_data, mode="train")
                datasets["eval_dataset"] = eval_dataset
            logger.info(f"datasets:{datasets}")
            # logger.info(train_dataset[0])
            train_kwargs.update(train_num=len(train_dataset), eval_num=len(eval_dataset))

        with LogCostContext(name="model train"):
            logger.info("building train args")
            train_args = get_train_args(**train_kwargs)
            logger.info("building callbacks")
            callbacks = get_callbacks(train_args=train_args, callback_config=callback_kwargs)
            logger.debug(f"train args:{train_args}")
            logger.debug(f"{callbacks=}")
            logger.info("do training")
            trainer = Trainer(
                model=self.nn_model,
                args=train_args,
                **datasets,
                data_collator=self.collator_fn,
                callbacks=callbacks
            )
            trainer.train()
