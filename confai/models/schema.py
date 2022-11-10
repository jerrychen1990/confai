#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: schema.py
@time: 2022/11/1 11:33
"""

from abc import ABC
from enum import Enum
from typing import Union, List, Tuple, Optional

from pydantic import BaseModel, Field, parse_obj_as, validator

from confai.decorator import adapt_batch

"""
基础数据类型
"""


class HashableModel(BaseModel):
    class Config:
        frozen = True
        allow_population_by_field_name = True

    def __eq__(self, other):
        return isinstance(other, self.__class__) and hash(other) == hash(self)


# 模型输入
class Example(HashableModel):
    extra: dict = Field(description="meta信息,用来和其他系统数据对接", default={})

    def get_ground_truth(self):
        raise NotImplementedError


# 模型输出
class Predict(HashableModel):
    pass


# 单文本模型输入
class TextExample(Example, ABC):
    text: str = Field(description="输入的文本")


# 文本对模型输入
class TextPairExample(Example, ABC):
    first_text: str = Field(description="第一段文本")
    second_text: str = Field(description="第二段文本")


# 标签
class Label(Predict):
    name: str = Field(description="标签名称")
    score: float = Field(description="标签概率", le=1., ge=0, default=1.)

    def __hash__(self):
        return hash(self.name)


Labels = List[Label]
# 单个标签或者多个标签，分别对应单标签分类任务和多标签分类任务
LabelOrLabels = Union[Labels, Label]


# 带标签的文本片段（比如NER任务的输出）
class TextSpan(Predict):
    text: str = Field(description="文本片段内容")
    span: Tuple[int, int] = Field(description="文本片段的下标区间，前闭后开")
    label: Label = Field(description="片段标签")


TextSpans = List[TextSpan]


class GenText(Predict):
    text: str = Field(description="生成的内容")
    score: float = Field(description="标签概率", le=1., ge=0, default=1.)


GenTexts = List[GenText]


# 文本分类模型输入
class TextClassifyExample(TextExample):
    label: Optional[LabelOrLabels] = Field(description="Ground Truth, 训练数据有此字段。根据label是不是list类型区分单标签任务和多标签任务")

    def get_ground_truth(self):
        return self.label


# 文本对模型输入
class TextPairClassifyExample(TextPairExample):
    label: Optional[LabelOrLabels] = Field(description="Ground Truth, 训练数据有此字段。训练数据有此字段。根据label是不是list类型区分单标签任务和多标签任务")

    def get_ground_truth(self):
        return self.label


# 文本片段分类模型输入
class TextSpanClassifyExample(TextExample):
    text_spans: Optional[TextSpans] = Field(description="Ground Truth， 训练数据有此字段")

    def get_ground_truth(self):
        return self.text_spans

    @validator("text_spans")
    def validate_atts(cls, v, values, field):
        for text_span in v:
            s, e = text_span.span
            if values["text"][s:e] != text_span.text:
                raise ValueError(f"span:{text_span.span} not match the text:{text_span.text}")
        return v


# text2text 模型输入
class TextGenExample(TextExample):
    gen: Optional[GenText] = Field(description="Ground Truth, 训练数据有此字段。")

    def get_ground_truth(self):
        return self.gen


"""
MOCK数据，方便直观了解schema格式
"""

MOCK_TEXT_CLASSIFY_EXAMPLE = {
    "text": "这是一篇内容不错的文章,但是文笔很差",
    "label": [
        {
            "name": "positive",
            "score": 1.
        }
    ]
}
MOCK_TEXT_CLASSIFY_PREDICT = [
    {
        "name": "positive",
        "score": .7
    },
    {
        "name": "negative",
        "score": .9
    }
]

MOCK_TEXT_PAIR_EXAMPLE = {
    "first_text": "你幸福么？",
    "second_text": "我姓曾"
}

MOCK_TEXT_PAIR_PREDICT = {
    "name": "不匹配"
}

MOCK_TEXT_SPAN_EXAMPLE = {
    "text": "小明出生在北京"
}

MOCK_TEXT_SPAN_PREDICT = [
    {
        "text": "小明",
        "span": [0, 2],
        "label": {
            "name": "PERSON",
            "score": 1.
        }
    },
    {
        "text": "北京",
        "span": [5, 7],
        "label": {
            "name": "LOCATION",
            "score": .87
        }
    }
]

MOCK_TEXT_SPELL_CORRECTION_EXAMPLE = {
    "text": "人工只能"
}

MOCK_TEXT_SPELL_CORRECTION_PREDICT = [
    {
        "word": "智",
        "index": 3,
        "score": 1.
    }
]

MOCK_TEXT_GEN_EXAMPLE = {
    "text": "白日依山尽"
}

MOCK_TEXT_GEN_PREDICT = {
    "text": "黄河入海流",
}


def build_union_instance(union_cls, data):
    for cls in union_cls.__args__:
        try:
            # logging.info(f"cls:{cls}, data:{data}")
            rs = cls(**data)
            return rs
        except Exception as e:
            # logging.exception(e)
            pass
    raise Exception("build union instance failed")


UnionPredict = Union[TextSpan, Label]

PathOrPaths = Union[str, List[str]]
DictOrDicts = Union[dict, List[dict]]
ExampleOrExamples = Union[Example, List[Example]]
PathOrDictOrExample = Union[PathOrPaths, DictOrDicts, ExampleOrExamples]
PredictOrPredicts = Union[Predict, List[Predict]]


@adapt_batch(ele_name="element")
def to_dict(element: Union[BaseModel, List[BaseModel]], **kwargs) -> DictOrDicts:
    return element.dict(**kwargs)


# 任务定义
class Task(Enum):
    # 文本分类
    TEXT_CLS = (TextClassifyExample, LabelOrLabels, MOCK_TEXT_CLASSIFY_EXAMPLE, MOCK_TEXT_CLASSIFY_PREDICT)
    # 文本对分类（可以适配文章分类，title和content分别作为first_text和second_text。也可以适配判断问答对是否匹配的二分类任务，question和answer分别作为first_text
    # 和second_text）
    TEXT_PAIR_CLS = (TextPairClassifyExample, LabelOrLabels, MOCK_TEXT_PAIR_EXAMPLE, MOCK_TEXT_PAIR_PREDICT)
    # 文本span抽取
    TEXT_SPAN_CLS = (TextSpanClassifyExample, TextSpans, MOCK_TEXT_SPAN_EXAMPLE, MOCK_TEXT_SPAN_PREDICT)
    # 文本生成
    TEXT_GEN = (TextGenExample, GenText, MOCK_TEXT_GEN_EXAMPLE, MOCK_TEXT_GEN_PREDICT)

    def __init__(self, input_cls, output_cls, mock_input, mock_output):
        self.input_cls = input_cls
        self.output_cls = output_cls
        self.mock_input = parse_obj_as(self.input_cls, mock_input)
        self.mock_output = parse_obj_as(self.output_cls, mock_output)
