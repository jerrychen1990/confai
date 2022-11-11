#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: output.py
@time: 2022/9/20 10:43
"""
import logging
from typing import Dict

from confai.evaluate import label2set, get_tp_fp_fn_set, get_unique_text_span
from confai.models.schema import *

logger = logging.getLogger(__name__)


#
# 将文本 类型结果输出
def get_text_classify_output(examples: List[TextClassifyExample],
                             preds: List[LabelOrLabels]) -> List[Dict]:
    rs_list = []
    assert len(examples) == len(preds)

    for example, pred in zip(examples, preds):
        rs_item = example.dict(exclude_none=True, exclude={"extra"})
        if isinstance(pred, list):
            rs_item.update(predict=[l.dict(exclude_none=True) for l in pred])
        else:
            rs_item.update(predict=pred.dict(exclude_none=True))

        if example.get_ground_truth() is not None:
            true_set = label2set(example.get_ground_truth())
            pred_set = label2set(pred)
            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        rs_list.append(rs_item)
    return rs_list


# 将sequence labeling结果输出
def get_text_span_classify_output(examples: List[TextSpanClassifyExample], preds: List[TextSpans]):
    output = []
    for example, pred in zip(examples, preds):
        rs_item = example.dict(exclude_none=True, exclude={"extra"})
        rs_item.update(predict=[e.dict(exclude_none=True) for e in pred])
        if example.get_ground_truth() is not None:
            true_set = set([get_unique_text_span(s) for s in example.get_ground_truth()])
            pred_set = set([get_unique_text_span(s) for s in pred])

            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        output.append(rs_item)
    return output


def get_text_spell_correction_output(examples, preds):
    output = []
    for example, pred in zip(examples, preds, exclude={"extra"}):
        rs_item = example.dict(exclude_none=True)
        rs_item.update(predict=[e.dict(exclude_none=True) for e in pred])
        if example.get_ground_truth() is not None:
            true_set = set([s.index for s in example.get_ground_truth()])
            pred_set = set([s.index for s in pred])
            tp_set, fp_set, fn_set = get_tp_fp_fn_set(true_set, pred_set)
            rs_item.update(tp_set=tp_set, fp_set=fp_set, fn_set=fn_set)
        output.append(rs_item)
    return output


def get_text_gen_output(examples: List[TextGenExample], preds: List[GenText]):
    output = []
    for example, pred in zip(examples, preds):
        rs_item = example.dict(exclude_none=True, exclude={"extra"})
        rs_item.update(predict=pred.dict(exclude_none=True))
        output.append(rs_item)
    return output


def get_output_func(task: Task):
    _task2output_func = {
        Task.TEXT_CLS: get_text_classify_output,
        Task.TEXT_SPAN_CLS: get_text_span_classify_output,
        Task.TEXT_GEN: get_text_gen_output
    }

    if task not in _task2output_func:
        raise ValueError(f"no output function found for task:{task}, valid tasks:"
                         f"{[e.name for e in _task2output_func.keys()]}")
    return _task2output_func[task]


def get_output_on_task(examples: List[Example], preds: List[PredictOrPredicts], task: Task) -> Dict:
    output_func = get_output_func(task)
    output = output_func(examples, preds)
    return output
