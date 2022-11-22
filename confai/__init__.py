#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: __init__.py.py
@time: 2022/11/1 10:41
"""

import os
import warnings
from logging.config import dictConfig

from confai.utils import read_config

__version__ = "0.0.1"


cur_path = os.path.dirname(os.path.abspath(__file__))
if "CONFAI_PATH" not in os.environ or True:
    confai_path = os.path.dirname(cur_path)
    os.environ["CONFAI_PATH"] = confai_path

logging_config_file_name = "logging.dev.yml" if eval(os.environ.get("CONFAI_DEBUG", "False")) else "logging.yml"
logging_config_file_name = os.path.join(cur_path, logging_config_file_name)
logging_config = read_config(logging_config_file_name, do_eval=False)
dictConfig(logging_config)

logging_dir = os.path.join(os.environ["CONFAI_PATH"], "logs")
if os.path.exists(logging_dir):
    os.makedirs(logging_dir)


# 过滤warning
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for .* type is zero.")
