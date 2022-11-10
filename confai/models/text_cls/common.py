#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: common.py
@time: 2022/11/3 11:20
"""
import logging
import os
from abc import ABC
from typing import Set

from snippets import dump_lines, load_lines, seq2dict, ensure_dir_path

from confai.models.core import ConfAIBaseModel
from confai.models.schema import Task

logger = logging.getLogger(__name__)


class BaseTextClassifyModel(ConfAIBaseModel, ABC):
    task = Task.TEXT_CLS

    def _load_config(self, config):
        super()._load_config(config)
        self.multi_label = self.task_config["multi_label"]
        self.max_len = self.task_config["max_len"]
        self.labels = load_lines(self.task_config['label_path'])
        self.label2id, self.id2label = seq2dict(self.labels)
        self.label_num = len(self.label2id)
        self.ignore_labels = self.task_config.get("ignore_labels", [])

    @ensure_dir_path
    def save_assets(self, path):
        dump_lines(self.labels, os.path.join(path, "labels.txt"))


def multi_label2id_vector(label_set: Set[str], label2id):
    """
    多标签，将token_label编码成分类的vector形式
    :param label_set:
    :param label2id:
    :return:
    """
    label_num = len(label2id)
    label_vec = [0] * label_num
    for label in label_set:
        label_vec[label2id[label]] = 1
    return label_vec
