#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: transformer_gen.py
@time: 2022/11/8 17:37
"""

import logging
from typing import Dict

import torch
from transformers import T5ForConditionalGeneration

from confai.models.schema import *
from confai.models.text_gen.common import BaseTextGenModel
from confai.models.torch_core import HFTorchModel, Feature

logger = logging.getLogger(__name__)


class TransformerGenModel(BaseTextGenModel, HFTorchModel):
    input_keys = ["input_ids", "attention_mask"]
    output_keys = ["logits"]
    nn_model_call = "generate"

    def example2feature(self, example: TextGenExample, mode: str) -> Dict[str, Feature]:
        feature = self.tokenizer(example.text, truncation=True, max_length=self.max_len)
        if mode == "train":
            decode_feature = self.tokenizer(example.gen.text, truncation=True,
                                            max_length=self.max_len)
            feature["labels"] = decode_feature["input_ids"]
        return feature

    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        max_label_len = max([len(f["labels"]) for f in features])
        labels = [f["labels"] + [0] * (max_label_len - len(f["labels"])) for f in features]
        labels = torch.tensor(labels)

        decoder_input_ids = self.nn_model.prepare_decoder_input_ids_from_labels(labels=labels)
        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> GenText:
        logger.debug(pred_features)
        sequence = pred_features["sequences"]

        pred_text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        return GenText(text=pred_text)

    def build_model(self, pretrained_model_name: str, **kwargs):
        local_pretrain_model_path = self.data_manager.get_local_path(pretrained_model_name)
        logger.info(f"initializing nn model with path:{local_pretrain_model_path}...")
        self.nn_model = T5ForConditionalGeneration.from_pretrained(local_pretrain_model_path)
        return self.nn_model

    def save_assets(self, path):
        pass
