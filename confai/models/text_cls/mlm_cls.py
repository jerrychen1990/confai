#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: mlm_cls.py
@time: 2022/11/15 18:02
"""
import copy
import logging
import random
import re
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from snippets import load_lines, jload
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import TruncationStrategy

import confai.models.functions as F
from confai.models.schema import *
from confai.models.text_cls.common import BaseTextClassifyModel
from confai.models.torch_core import HFTorchModel, Feature
from confai.utils import inverse_dict

logger = logging.getLogger(__name__)


def get_mask_num(tokenizer: PreTrainedTokenizerBase, text: str):
    ids = tokenizer.encode(text)
    masks = [e for e in ids if e == tokenizer.mask_token_id]
    return len(masks)


class MLMCLSModel(BaseTextClassifyModel, HFTorchModel):
    input_keys = ["input_ids", "token_type_ids", "attention_mask"]
    output_keys = ["logits"]

    def __init__(self, config):
        super().__init__(config=config)
        self.prompts = load_lines(self.task_config["prompt_path"])
        mask_nums = [get_mask_num(self.tokenizer, e) for e in self.prompts]
        assert max(mask_nums) == min(mask_nums), f"max mask num:{max(mask_nums)} and min mask num {min(mask_nums)} " \
                                                 f"in prompts is not the same!"
        self.mask_num = mask_nums[0]
        self._init_token2label()

    def _init_token2label(self):
        self.word2label = jload(self.task_config["token2label_path"])
        self.label2words = inverse_dict(self.word2label)
        self.token2label, self.label2token = defaultdict(lambda: defaultdict(set)), defaultdict(list)
        for word, label in self.word2label.items():
            if label not in self.labels:
                logger.warning(f"invalid label:{label} in {self.task_config['token2label_path']}, skip it!")
                continue
            tokens = self.tokenizer.tokenize(word, add_special_tokens=False)
            if len(tokens) != self.mask_num:
                logger.warning(f"invalid tokens:{tokens}, not fit the mask num:{self.mask_num} of prompt, skip it!")
                continue
            for idx, token in enumerate(tokens):
                self.token2label[idx][token].add(label)
            self.label2token[label].append(tuple(tokens))
        self.valid_tokens = list(set([token for token_dict in self.token2label.values()
                                      for token in token_dict.keys()]))
        self.valid_token_ids = self.tokenizer.convert_tokens_to_ids(self.valid_tokens)
        self.token2id = dict(zip(self.valid_tokens, self.valid_token_ids))
        self.id2token = dict(zip(self.valid_token_ids, self.valid_tokens))

    def example2feature(self, example: TextClassifyExample, mode: str) -> Dict[str, Feature]:
        prompt = random.choice(self.prompts).replace("[X]", "")
        text = example.text
        tokenize_kwargs = dict(text=text, text_pair=prompt, truncation=TruncationStrategy.ONLY_FIRST,
                               max_length=self.max_len)

        if mode == "train":
            label_name = example.label.name
            word = random.choice(self.label2words[label_name])
            text_target = text
            text_pair_target = re.sub(pattern='(\[MASK\])+', repl=word, string=prompt)
            tokenize_kwargs.update(text_target=text_target, text_pair_target=text_pair_target)
        feature = self.tokenizer(**tokenize_kwargs)
        # tokens = self.tokenizer.convert_ids_to_tokens(feature["input_ids"])
        # logger.debug(tokens)
        # mask_idxs = [idx for idx, t in enumerate(tokens) if t == self.tokenizer.mask_token]
        # feature.update(prompt_text=text, tokens=tokens, mask_idxs=mask_idxs)
        return feature

    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        max_len = batch["input_ids"].shape[1]
        labels = [e["labels"] + [0] * (max_len - len(e["labels"])) for e in features]
        # logger.info(labels)
        labels = torch.tensor(labels)
        batch["labels"] = labels

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> LabelOrLabels:
        logits = pred_features["logits"]
        # logger.debug(f"{logits.shape=}")
        input_ids = features["input_ids"]
        mask_idxs = [idx for idx, t in enumerate(input_ids) if t == self.tokenizer.mask_token_id]
        # logger.debug(f"{mask_idxs=}")
        # logger.debug(f"{self.valid_token_ids=}")
        valid_logits = logits[mask_idxs][:, self.valid_token_ids]
        # logger.debug(f"{valid_logits=}")
        label2logit = defaultdict(int)
        for idx, logits in enumerate(valid_logits):
            assert len(self.valid_tokens) == len(logits)
            for token, logit in zip(self.valid_tokens, logits):
                for label in self.token2label[idx].get(token, []):
                    # logger.debug(f"add {logit} to {label} for {idx=}, {token=}")

                    label2logit[label] += logit
        logger.debug(f"{label2logit=}")
        label2logit = list(label2logit.items())
        probs = F.softmax([e[1] for e in label2logit])
        logger.debug(f"{probs=}")
        max_label_idx = np.argmax(probs)
        return Label(name=label2logit[max_label_idx][0], score=probs[max_label_idx])

    def build_model(self, pretrained_model_name, **kwargs):
        local_pretrain_model_path = self.data_manager.get_local_path(pretrained_model_name)
        logger.info(f"initializing nn model with path:{local_pretrain_model_path}...")
        self.nn_model = AutoModelForMaskedLM.from_pretrained(local_pretrain_model_path)
        return self.nn_model
