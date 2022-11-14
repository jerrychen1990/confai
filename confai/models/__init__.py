#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: __init__.py.py
@time: 2022/11/1 11:33
"""
from snippets import jload

from confai.models.core import ConfAIBaseModel
from confai.models.nn_core import NNModelConfig
from confai.models.text_cls import CLSTokenClsModel
from confai.models.text_gen import TransformerGenModel
from confai.models.text_span_cls import SeqLabelingModel

_all_models = [CLSTokenClsModel] + [TransformerGenModel] + [SeqLabelingModel]

__name2model_cls = {model_cls.__name__: model_cls for model_cls in _all_models}


# get model class by name
def get_model_cls(model_name: str) -> type(ConfAIBaseModel):
    if model_name not in __name2model_cls:
        raise ValueError(f"invalid model name: {model_name}, valid model names: {list(__name2model_cls.keys())}")
    return __name2model_cls[model_name]


def load_model(path=None, config: NNModelConfig = None, **kwargs) -> ConfAIBaseModel:
    if not config:
        if path:
            config = NNModelConfig.parse_obj(jload(ConfAIBaseModel.get_config_path(path)))
        else:
            raise ValueError("neither path or config is given!")

    model_cls = get_model_cls(config.model_cls)
    return model_cls.load(path=path, config=config, **kwargs)
