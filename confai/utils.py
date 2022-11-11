#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: utils.py
@time: 2022/11/1 15:31
"""
import copy
import random
import re
from configparser import ConfigParser
from dataclasses import fields
from typing import Union

import yaml
from snippets.decorators import *
from snippets.utils import *

logger = logging.getLogger(__name__)


# apply function to one element or batch of elements
def batch_apply(func, ele, **kwargs):
    if isinstance(ele, list):
        return [func(ele=e, **kwargs) for e in ele]
    else:
        return func(ele=ele, **kwargs)


# convert list of dicts to dict with list values
def dlist2ldict(d_list: List[Dict]) -> Dict[Any, List]:
    keys = d_list[0].keys
    rs = {k: list() for k in keys()}
    for e in d_list:
        for k, v in e.items():
            rs[k].append(v)
    return rs


# convert dict with list values to list of dicts
def ldict2dlist(ldict: Dict[Any, Union[List, np.ndarray]]) -> List[Dict]:
    l = len(list(ldict.values())[0])
    rs = []
    for idx in range(l):
        d = {k: v[idx] for k, v in ldict.items()}
        rs.append(d)
    return rs


# 安全的调用函数， 讲kwargs中不在函数参数列表中的部分去除掉之后再调用
def safe_call(func, *args, **kwargs):
    valid_keys = func.__annotations__.keys()
    # logger.info(kwargs)
    # logger.info(valid_keys)
    valid_kwargs = {k: kwargs[k] for k in valid_keys if k in kwargs}
    # logger.info(args)
    # logger.info(valid_kwargs)
    return func(*args, **valid_kwargs)


# build data class with kwargs safely
def safe_build_data_cls(cls: type, kwargs):
    _fields = fields(cls)
    valid_keys = set(e.name for e in _fields)
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return cls(**kwargs)


def eval_env(text):
    pattern = "\$\{.*?\}"
    for item in re.findall(pattern, text):
        text = text.replace(item, os.environ[item[2:-1]])
    return text


# # 深度遍历用u更新d
def deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# 随机拆分list
# pct_or_num如果是0-1的浮点数, 按照1-pct_or_num:pct_or_num拆分
# pct_or_num如果是整数, 按照len(l)-pct_or_num:pct_or_num 拆分
def random_split_list(l: List, pct_or_num: Union[float, int]):
    tmp = copy.copy(l)
    random.shuffle(tmp)
    if isinstance(pct_or_num, float) and 0. < pct_or_num < 1.:
        sample_num = int(len(l) * pct_or_num)
    else:
        sample_num = int(pct_or_num)

    return tmp[sample_num:], tmp[:sample_num]


def jload_multi_files(files):
    rs = []
    for file in files:
        rs.extend(jload_lines(file))
    return rs


# 读取配置文件，支持.json/.ini/.yaml格式
# 可以继承另一个配置文件
# 可以引入环境变量
def read_config(config_path: str, do_eval=True, parse_env=True) -> dict:
    def eval_param(param):
        if isinstance(param, str):
            if parse_env:
                param = eval_env(param)
            if do_eval:
                try:
                    if param.upper() == "TRUE":
                        return True
                    if param.upper() == "FALSE":
                        return False
                    param = eval(param)
                except:
                    pass
            return param

        if isinstance(param, dict):
            return {k: eval_param(v) for k, v in param.items()}
        if isinstance(param, list):
            return [eval_param(v) for v in param]
        return param

    # convert cfg data to dict
    def cfg2dict(cfg):
        sections = cfg.sections()
        rs = {k: dict(cfg[k]) for k in sections}
        return rs

    cfg_dict = dict()
    if not os.path.exists(config_path):
        raise Exception(f"file {config_path} not exists!")

    logger.info(f"parsing config with path:{config_path}")
    if config_path.endswith(".ini"):
        parser = ConfigParser()
        parser.read(config_path)
        cfg_dict = cfg2dict(parser)
    elif config_path.endswith(".json"):
        cfg_dict = jload(config_path)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, mode='r', encoding="utf") as stream:
            cfg_dict = yaml.safe_load(stream)
    else:
        raise ValueError(f"invalid config path:{config_path}")
    cfg_dict = eval_param(cfg_dict)

    if cfg_dict.get("common_config", {}).get("base_config"):
        logger.info("loading base config...")
        base_cfg_dict = read_config(cfg_dict["common_config"]["base_config"])
    else:
        base_cfg_dict = dict()
    deep_update(base_cfg_dict, cfg_dict)
    return base_cfg_dict
