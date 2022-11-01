#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: decorator.py
@time: 2022/11/1 11:44
"""

from snippets.decorators import *


# adapt function with batch data
def adapt_batch(ele_name):
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            ele = kwargs.pop(ele_name, None)
            if ele is None:
                return func(*args, **kwargs)
            # print(args)
            # print(kwargs)
            # print(ele)
            if isinstance(ele, list):
                return [func(*args, **{ele_name: e}, **kwargs) for e in ele]
            else:
                return func(*args, **{ele_name: ele}, **kwargs)

        return wrapped

    return wrapper
