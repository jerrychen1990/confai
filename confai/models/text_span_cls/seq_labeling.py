#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: seq_labeling.py
@time: 2022/11/14 15:33
"""

import logging
from collections import namedtuple
from enum import unique, Enum
from typing import Dict
from typing import List, Tuple
import torch.nn.functional as F

import numpy as np
import torch
from snippets import seq2dict
from transformers import AutoModelForTokenClassification

from confai.models.nn_core import Feature, NNModel
from confai.models.torch_core import HFTorchModel
from confai.models.text_span_cls.common import BaseTextSpanClassifyModel, get_char2token
from confai.models.schema import TextSpanClassifyExample, TextSpans, TextSpan, Label, PathOrDictOrExample, \
    PredictOrPredicts, Example

logger = logging.getLogger(__name__)

LABEL_SEP = "_"


@unique
# 所有的序列标注编码、解码格式BIO/BIOES等等
class SeqLabelStrategy(Enum):
    # BIO编码
    BIO = ("B", "I", None, None, "O", False, False)
    # BIO编码，只有B后面带有span的类型，I不区分span类别
    BIO_SIMPLE = ("B", "I", None, None, "O", False, True)
    # BIOES编码
    BIOES = ("B", "I", "E", "S", "O", False, False)
    # BIOES编码, 只有B和S后面带有span的类型，IEO不区分span的类别
    BIOES_SIMPLE = ("B", "I", "E", "S", "O", False, True)
    # 用B和E两个指正标注span的范围
    BE_POINTER = ("B", None, "E", None, None, True, False)

    def __init__(self, begin, mid, end, single, empty, is_pointer=False, simple_mode=False):
        self.begin = begin
        self.mid = mid
        self.end = end
        self.single = single
        self.empty = empty
        self.is_pointer = is_pointer
        self.simple_mode = simple_mode
        self.no_empty_list = [e for e in [self.begin, self.mid, self.end, self.single] if e]
        self.start_set = set([e for e in [begin, single] if e])
        self.end_set = set([e for e in [end, single] if e])
        self.mid_set = {mid}

    def get_single(self):
        return self.single if self.single else self.begin

    def get_end(self):
        return self.end if self.end else self.mid


LabelInfo = namedtuple("LabelInfo", ["part", "name", "score"])


# 将label_type和label_part组合成一个token_label。比如"B"+"人物",编码得到"B_人物"
def encode_label(label_part: str, label_type: str, seq_label_strategy: SeqLabelStrategy) -> str:
    if seq_label_strategy.simple_mode and label_part not in seq_label_strategy.start_set:
        return label_part
    else:
        return LABEL_SEP.join([label_part, label_type])


# 用和encode_label相同的方法，将token的label解码得到将label_type和label_part。比如"B_人物"解码得到"B"+"人物"
def decode_label(label: str) -> Tuple[str, str]:
    fields = label.split(LABEL_SEP)
    label_part = fields[0]
    label_type = LABEL_SEP.join(fields[1:]) if len(fields) > 1 else None
    return label_part, label_type


def apply_seq_label_strategy(label_list: List[str], seq_label_strategy: SeqLabelStrategy) -> List[str]:
    full_label_list = [] if seq_label_strategy.is_pointer else [seq_label_strategy.empty]
    for label in label_list:
        for label_part in seq_label_strategy.no_empty_list:
            full_label = encode_label(label_part, label, seq_label_strategy)
            if full_label not in full_label_list:
                full_label_list.append(full_label)
    return full_label_list


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的mid_label
def is_mid(start_label_type, label_part, label_type,
           span_extract_strategy: SeqLabelStrategy):
    if not span_extract_strategy.simple_mode:
        if start_label_type != label_type:
            return False
    if label_part == span_extract_strategy.mid:
        return True
    return False


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的明确end_label
def is_exact_end(start_label_type, label_part, label_type,
                 span_extract_strategy: SeqLabelStrategy):
    if not span_extract_strategy.simple_mode:
        if start_label_type != label_type:
            return False
    return label_part == span_extract_strategy.end


# 判断一个label在span_extract_strategy编码方式下是不是对应start_label的可能的end_label
def is_valid_end(label_part, gap, span_extract_strategy: SeqLabelStrategy):
    if gap == 0 and label_part == span_extract_strategy.get_single():
        return True
    if gap > 0 and label_part == span_extract_strategy.get_end():
        return True
    return False


# 给定一个不重叠的labeled token序列，以及span的start位置、label。找到符合span_extract_strategy规范的最长span的范围
def get_valid_span(label_infos: List[LabelInfo], start_idx, start_label_type,
                   span_extract_strategy: SeqLabelStrategy):
    pre_idx = start_idx
    for idx in range(start_idx + 1, len(label_infos)):
        label_part, label_type, score = label_infos[idx]
        if is_exact_end(start_label_type, label_part, label_type, span_extract_strategy):
            return start_idx, idx + 1
        if is_mid(start_label_type, label_part, label_type, span_extract_strategy):
            continue
        pre_idx = idx - 1
        break
    pre_part, _, _ = label_infos[pre_idx]
    if is_valid_end(pre_part, pre_idx - start_idx, span_extract_strategy):
        return start_idx, pre_idx + 1
    return None


# 给定一个不重叠的labeled token list, 根据span_extract_strategy 解码出所有token级别的span范围和类别
def get_valid_spans(label_infos: List[LabelInfo], span_extract_strategy: SeqLabelStrategy):
    rs_list = []
    for idx, (label_part, label_name, score) in enumerate(label_infos):
        if label_part == span_extract_strategy.single:
            rs_list.append((label_name, (idx, idx + 1)))
        elif label_part == span_extract_strategy.begin:
            valid_span = get_valid_span(label_infos, idx, label_name, span_extract_strategy)
            if valid_span:
                rs_list.append((label_name, valid_span))
    rs_list = sorted(rs_list, key=lambda x: x[1])
    return rs_list


# 根据原始数据以及span_list解码出SeqSpan结构的数据
def decode_text_spans(text, offset_mapping, spans: List[Tuple[str, Tuple[int, int]]],
                      label_infos: List[LabelInfo]) -> TextSpans:
    # logger.info(spans)
    text_spans = []

    for label_type, (start, end) in spans:
        end = end - 1
        # 计算开头的score
        score = label_infos[start].score
        if 0 <= start <= end < len(offset_mapping) and \
                offset_mapping[start] and offset_mapping[end]:
            char_start, _ = offset_mapping[start]
            _, char_end = offset_mapping[end]
            span_text = text[char_start: char_end]
            # logger.info(span_text)
            if span_text:
                text_spans.append(TextSpan(text=span_text, label=Label(name=label_type, score=score),
                                           span=(char_start, char_end)))
    return text_spans


# 给定原始数据以及不重叠的token级别label list，解码出SeqSpan的list
def decode_token_label_sequence(text, offset_mapping, label_infos: List[LabelInfo],
                                span_extract_strategy: SeqLabelStrategy) -> TextSpans:
    valid_spans = get_valid_spans(label_infos, span_extract_strategy)
    text_spans = decode_text_spans(text, offset_mapping, valid_spans, label_infos)
    # logger.info(text_spans)
    return text_spans


# 给定token list以及不重叠的char级别的span信息，得到token级别的标注结果
def get_token_label_sequence(token_len, text_spans: TextSpans, char2token, seq_label_strategy: SeqLabelStrategy):
    if seq_label_strategy.is_pointer:
        raise Exception(f"pointer strategy only work with multi_label sequence labeling task")
    tokens_label_sequence = [seq_label_strategy.empty for _ in range(token_len)]

    def is_all_empty(s, e):
        for i in range(s, e + 1):
            if tokens_label_sequence[i] != seq_label_strategy.empty:
                return False
        return True

    for text_span in text_spans:
        text, label, (start, end) = text_span.text, text_span.label.name, text_span.span
        end = end - 1
        token_start, token_end = char2token[start], char2token[end]
        if is_all_empty(token_start, token_end):
            if token_end == token_start:
                tokens_label_sequence[token_start] = encode_label(seq_label_strategy.get_single(),
                                                                  label, seq_label_strategy)
            else:
                tokens_label_sequence[token_start] = encode_label(seq_label_strategy.begin,
                                                                  label, seq_label_strategy)
                for idx in range(token_start + 1, token_end):
                    tokens_label_sequence[idx] = encode_label(seq_label_strategy.mid, label,
                                                              seq_label_strategy)
                tokens_label_sequence[token_end] = encode_label(seq_label_strategy.get_end(),
                                                                label, seq_label_strategy)
    return tokens_label_sequence


class SeqLabelingModel(BaseTextSpanClassifyModel, HFTorchModel):
    input_keys = ["input_ids", "token_type_ids", "attention_mask"]
    output_keys = ["logits"]

    def _load_config(self, config):
        super()._load_config(config)
        self.max_len = self.task_config["max_len"]
        self.seq_label_strategy: SeqLabelStrategy = SeqLabelStrategy[
            self.task_config['seq_label_strategy']]
        self.multi_label = self.task_config.get("multi_label", False)

        self.token_labels = apply_seq_label_strategy(self.labels, self.seq_label_strategy)
        self.label2id, self.id2label = seq2dict(self.token_labels)
        self.label_num = len(self.label2id)

    def example2feature(self, example: TextSpanClassifyExample, mode: str) -> Dict[str, Feature]:
        features = self.tokenizer(example.text, truncation=True, return_offsets_mapping=True, max_length=self.max_len)
        features.update(text=example.text)
        if mode == "train":
            char2token = get_char2token(example.text, features["offset_mapping"])
            tokens = self.tokenizer.convert_ids_to_tokens(features["input_ids"])
            ner_tag_labels = get_token_label_sequence(token_len=len(features["input_ids"]),
                                                      text_spans=example.text_spans,
                                                      char2token=char2token,
                                                      seq_label_strategy=self.seq_label_strategy)
            ner_tags = [self.label2id[l] for l in ner_tag_labels]
            features.update(labels=ner_tags, tokens=tokens)
        return features

    def _update_batch(self, batch: Dict[str, torch.Tensor], features: List[Dict[str, Feature]]):
        tgt_length = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            label = torch.Tensor(f["labels"])
            # logger.debug(f"{label.shape}")
            # logger.debug(f"{label.dtype}")
            gap = tgt_length - label.shape[0]
            label = F.pad(label, (0, gap)).type(torch.int64)
            # logger.debug(f"{label.dtype}")
            labels.append(label)
        labels = torch.stack(labels)
        batch["labels"] = labels

    def feature2predict(self, features: Dict[str, Feature], pred_features: Dict[str, Feature]) -> TextSpans:
        logits = pred_features["logits"]
        label_ids = np.argmax(logits, axis=-1)
        a = np.exp(logits)
        b = np.expand_dims(np.exp(logits).sum(axis=-1), axis=-1)
        label_probs = a / b

        label_infos = [LabelInfo(*decode_label(self.id2label[i]), label_probs[idx][i]) for idx, i in enumerate(
            label_ids)]
        label_infos = label_infos[:len(features["input_ids"])]
        text_spans = decode_token_label_sequence(text=features["text"], offset_mapping=features["offset_mapping"],
                                                 label_infos=label_infos,
                                                 span_extract_strategy=self.seq_label_strategy)
        return text_spans

    def build_model(self, pretrained_model_name: str, **kwargs):
        pretrain_path = self.data_manager.get_local_path(pretrained_model_name)
        logger.info(f"initializing nn model with path:{pretrain_path}...")
        self.nn_model = AutoModelForTokenClassification.from_pretrained(pretrain_path, id2label=self.id2label)
        return self.nn_model
