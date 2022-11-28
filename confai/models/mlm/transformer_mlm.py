#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: transformer_mlm.py
@time: 2022/11/28 14:36
"""
import copy
import logging
import random
from typing import Dict

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

from confai.models.functions import softmax
from confai.models.mlm.common import BaseMLMModel
from confai.models.torch_core import HFTorchModel, Feature
from confai.schema import *

logger = logging.getLogger(__name__)


class TransformerMLMModel(BaseMLMModel, HFTorchModel):
    input_keys = ["input_ids", "token_type_ids", "attention_mask"]
    output_keys = ["logits"]

    def __init__(self, config):
        super(TransformerMLMModel, self).__init__(config=config)
        self.mask_pct = self.task_config.get("mask_pct", 0.15)

    def _random_mask(self, feature: Dict[str, Feature]):
        for idx, token in enumerate(feature["input_ids"]):
            rnd = np.random.random()
            logger.debug(f"{idx}, {rnd}, {random.random()}")
            if rnd < self.mask_pct:
                feature["input_ids"][idx] = self.tokenizer.mask_token_id

    def example2feature(self, example: MLMExample, mode: str) -> Dict[str, Feature]:
        feature = self.tokenizer(example.text, truncation=True, max_length=self.max_len)
        masked_idxs = [idx for idx, token in enumerate(feature["input_ids"]) if token == self.tokenizer.mask_token_id]
        labels = []
        if mode == "train":
            masked_tokens = example.get_ground_truth()
            if masked_tokens:
                if not masked_idxs or len(masked_idxs) != len(masked_tokens):
                    raise ValueError(f"masked_idxs:{masked_idxs}, masked_tokens:{masked_tokens}, not match!")
                else:
                    labels = copy.copy(feature["input_ids"])
                    for idx, token in zip(masked_idxs, masked_tokens):
                        labels[idx] = self.tokenizer.convert_tokens_to_ids(token.word)
            else:
                if masked_idxs:
                    raise ValueError(f"masked token in input, but no ground truth!")
                else:
                    labels = copy.copy(feature["input_ids"])
                    self._random_mask(feature)
                    masked_idxs = [idx for idx, token in enumerate(feature["input_ids"])
                                   if token == self.tokenizer.mask_token_id]
                    logger.debug(f"masked_idxs:{masked_idxs}")

        feature.update(masked_idxs=masked_idxs, labels=labels)
        return feature

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> Tokens:
        masked_idxs = features["masked_idxs"]
        logits = pred_features["logits"][masked_idxs]
        logger.debug(logits)
        probs = softmax(logits)
        token_ids = np.argmax(probs, axis=1)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        probs = [p[idx] for idx, p in zip(token_ids, probs)]

        logger.debug(probs)
        logger.debug(tokens)
        tokens = [Token(word=token, score=prob)
                  for token, prob in zip(tokens, probs)]

        return tokens

    def _do_build_model(self, pretrained_model_name, **kwargs):
        local_pretrain_model_path = self.data_manager.get_local_path(pretrained_model_name)
        logger.info(f"initializing nn model with path:{local_pretrain_model_path}...")
        self.nn_model = AutoModelForMaskedLM.from_pretrained(local_pretrain_model_path)
        return self.nn_model

    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        max_len = batch["input_ids"].shape[1]
        labels = [e["labels"] + [0] * (max_len - len(e["labels"])) for e in features]
        # logger.info(labels)
        labels = torch.tensor(labels)
        batch["labels"] = labels
