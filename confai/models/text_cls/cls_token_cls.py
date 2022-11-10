#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: cls_token_cls.py
@time: 2022/11/3 11:18
"""
import logging
from typing import Dict

import numpy as np
import torch
from numpy import ndarray
from transformers import AutoModelForSequenceClassification

from confai.models.torch_core import HFTorchModel, Feature
from confai.models.schema import *
from confai.models.text_cls.common import BaseTextClassifyModel

logger = logging.getLogger(__name__)


class CLSTokenClsModel(BaseTextClassifyModel, HFTorchModel):
    input_keys = ["input_ids", "token_type_ids", "attention_mask"]
    output_keys = ["logits"]

    def __init__(self, config):
        super().__init__(config=config)
        self.max_len = self.task_config["max_len"]

    def example2feature(self, example: Example, mode: str) -> Dict[str, Feature]:
        feature = self.tokenizer(example.text, truncation=True, max_length=self.max_len)
        if mode == "train":
            if self.multi_label:
                onehot_label = [0.] * self.label_num
                for l in example.label:
                    idx = self.label2id[l.name]
                    onehot_label[idx] = 1.
                feature["label"] = onehot_label

            else:
                feature["label"] = self.label2id[example.label.name]
            return feature

        return feature

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> PredictOrPredicts:
        logger.debug(pred_features)
        logits: ndarray = pred_features["logits"]
        if self.multi_label:
            labels = []
            # logger.info(logits)
            probs = 1 / (1 + np.exp(-logits))
            logger.debug(probs)
            # exp_logits = np.exp(logits)
            # probs = exp_logits / sum(exp_logits)
            for idx, prob in enumerate(probs):
                if prob >= 0.5:
                    labels.append(Label(name=self.id2label[idx], score=prob))
            return labels
        else:
            label_id = np.argmax(logits)
            # logger.info(f"label_id:{label_id}")
            exp_logits = np.exp(logits)
            probs = exp_logits / sum(exp_logits)
            # logger.info(f"probs:{probs}")
            score = probs[label_id]
            label = Label(name=self.id2label[label_id], score=score)
            return label

    def build_model(self, pretrained_model_name, **kwargs):
        local_pretrain_model_path = self.data_manager.get_local_path(pretrained_model_name)
        logger.info(f"initializing nn model with path:{local_pretrain_model_path}...")

        problem_type = "multi_label_classification" if self.multi_label else "single_label_classification"

        self.nn_model = AutoModelForSequenceClassification.from_pretrained(local_pretrain_model_path,
                                                                           id2label=self.id2label,
                                                                           problem_type=problem_type)

        return self.nn_model

    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        labels = [e["label"] for e in features]
        # logger.info(labels)
        labels = torch.tensor(labels)
        batch["labels"] = labels
