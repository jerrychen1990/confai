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
