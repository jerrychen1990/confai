#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_utils.py
@time: 2022/11/3 11:06
"""

import unittest
from confai.utils import *

# current absolute path
cur_path = os.path.abspath(os.path.dirname(__file__))


class TestUtils(unittest.TestCase):

    def test_read_config(self):
        path = os.path.join(cur_path, "../conf/examples/base.ini")
        config = read_config(path)
        logger.info(jdumps(config))

    def test_random_split(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        l1, l2 = random_split_list(l, 0.1)
        self.assertEqual(1, len(l2))
        self.assertEqual(9, len(l1))

        l1, l2 = random_split_list(l, 6)
        self.assertEqual(6, len(l2))
        self.assertEqual(4, len(l1))
