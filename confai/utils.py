#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: utils.py
@time: 2022/11/1 15:31
"""
from typing import List, Dict, Any, Union

import numpy as np


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
