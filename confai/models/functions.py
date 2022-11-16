#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: functions.py
@time: 2022/11/16 11:08
"""
import numpy as np


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x)
    x = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
    return x
