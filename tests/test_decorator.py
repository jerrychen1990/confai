#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_decorator.py
@time: 2022/11/1 14:54
"""
import unittest
from confai.decorator import *


class TestUtils(unittest.TestCase):
    def test_adapt_single(self):
        @adapt_batch(ele_name="ele")
        def add(b, ele, **kwargs):
            return ele+b

        batch_rs = add(1, ele=[1, 2, 3])
        self.assertEqual([2, 3, 4], batch_rs)

        single_rs = add(ele=1, b=4)
        self.assertEqual(5, single_rs)

        single_rs = add(1, 4, data=[3])
        self.assertEqual(5, single_rs)

