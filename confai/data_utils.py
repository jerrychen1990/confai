#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: data_utils.py
@time: 2022/11/1 15:39
"""
from abc import abstractmethod
import logging
import os
from abc import abstractmethod
from typing import Callable, Type
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AbstractDataManager:
    HOME_ENV = None

    @classmethod
    @abstractmethod
    def get_tokenizer(cls, tokenizer_path: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def collate(cls, features: List[Dict[str, List]], tokenizer, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def get_local_path(cls, path: str):
        if path.startswith("/"):
            return path
        if cls.HOME_ENV in os.environ:
            local_path = os.path.join(os.environ[cls.HOME_ENV], path)
            if os.path.exists(local_path):
                return local_path
            return path
        return path

    @classmethod
    @abstractmethod
    def save_tokenizer(cls, tokenizer, path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_dataset(cls, data_or_path: DataOrPath, task: Task, map_fn: Callable):
        raise NotImplementedError
