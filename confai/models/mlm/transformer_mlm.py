#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: transformer_mlm.py
@time: 2022/11/28 14:36
"""
import copy
import logging
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

    def _random_mask(self, feature: Dict[str, Feature]) -> List[int]:
        mask_idxs = []

        for idx, token in enumerate(feature["input_ids"]):
            rnd = np.random.random()
            if rnd < self.mask_pct:
                if rnd < self.mask_pct * 0.8:
                    feature["input_ids"][idx] = self.tokenizer.mask_token_id
                elif rnd < self.mask_pct * 0.9:
                    pass
                else:
                    feature["input_ids"][idx] = np.random.choice(self.tokenizer.vocab_size)
                mask_idxs.append(idx)
        return mask_idxs

    def example2feature(self, example: MLMExample, mode: str) -> Dict[str, Feature]:
        feature = self.tokenizer(example.text, truncation=True, max_length=self.max_len)
        mask_idxs = [idx for idx, token in enumerate(feature["input_ids"]) if token == self.tokenizer.mask_token_id]
        if mode == "train":
            labels = [-100] * len(feature["input_ids"])
            mask_tokens = example.get_ground_truth()
            if mask_tokens:
                if not mask_idxs or len(mask_idxs) > len(mask_tokens):
                    raise ValueError(f"mask_idxs:{mask_idxs} is longer than mask_tokens:{mask_tokens}!")
                else:
                    for idx, token in zip(mask_idxs, mask_tokens):
                        labels[idx] = self.tokenizer.convert_tokens_to_ids(token.word)
            else:
                if mask_idxs:
                    raise ValueError(f"mask token in input, but no ground truth!")
                else:
                    tgt_ids = copy.copy(feature["input_ids"])
                    mask_idxs = self._random_mask(feature)
                    for idx in mask_idxs:
                        labels[idx] = tgt_ids[idx]
            feature.update(labels=labels)
        feature.update(mask_idxs=mask_idxs)
        return feature

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> Tokens:
        mask_idxs = features["mask_idxs"]
        logits = pred_features["logits"][mask_idxs]
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
        labels = [e["labels"] + [-100] * (max_len - len(e["labels"])) for e in features]
        # logger.info(labels)
        labels = torch.tensor(labels)
        batch["labels"] = labels
