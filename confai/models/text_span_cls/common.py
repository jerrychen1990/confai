#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: common.py
@time: 2022/11/14 15:34
"""
import logging
import os
from abc import ABC

from snippets import dump_lines, load_lines, ensure_dir_path

from confai.models.core import ConfAIBaseModel, ModelConfig
from confai.models.text_cls.common import multi_label2id_vector
from confai.models.schema import Task

logger = logging.getLogger(__name__)


class BaseTextSpanClassifyModel(ConfAIBaseModel, ABC):
    task = Task.TEXT_SPAN_CLS

    def _load_config(self, config:ModelConfig):
        super()._load_config(config)
        self.max_len = self.task_config['max_len']
        self.multi_label = self.task_config.get("multi_label", False)
        self.labels = load_lines(self.task_config['label_path'])

    @ensure_dir_path
    def save_assets(self, path):
        dump_lines(self.labels, os.path.join(path, "labels.txt"))


def get_char2token(text, offset_map):
    char2token = [-1] * len(text)
    for idx, (s, e) in enumerate(offset_map):
        for i in range(s, e):
            char2token[i] = idx
    return char2token


# 将token_label编码成分类的vector形式
def token_label2classify_label_input(target_token_label_sequence, multi_label, label2id):
    if multi_label:
        classify_label_input = [multi_label2id_vector(e, label2id) for e in target_token_label_sequence]
    else:
        classify_label_input = [label2id[e] for e in target_token_label_sequence]

    return classify_label_input
