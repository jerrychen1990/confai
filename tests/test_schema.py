#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: test_schema.py
@time: 2022/11/1 15:10
"""

import unittest
import logging

from snippets import jdumps

from confai.models.schema import *

logger = logging.getLogger(__name__)


class TestSchema(unittest.TestCase):

    # 展示每个task的信息
    def test_tasks(self):
        for task in Task:
            logger.info(f"visiting task:{task}")
            logger.info(f"input_cls:{task.input_cls}")
            logger.info(f"mock_input_instance:{task.mock_input}")
            logger.info(f"mock_input_data:{to_dict(element=task.mock_input, exclude_unset=True)}")

            logger.info(f"output_cls:{task.output_cls}")
            logger.info(f"mock_output_instance:{task.mock_output}")
            logger.info(f"mock_output_data:{to_dict(element=task.mock_output, exclude_unset=True)}")

            logger.info("*" * 40)

    # 测试pydantic和dict的转化，如果格式不对，会抛出异常
    def test_data_convert(self):
        raw_data = MOCK_TEXT_CLASSIFY_EXAMPLE
        logger.info("raw data:")
        logger.info(jdumps(raw_data))
        pydantic_instance = parse_obj_as(TextClassifyExample, raw_data)
        logger.info("pydantic instance:")
        logger.info(pydantic_instance)
        parsed_raw_data = to_dict(element=pydantic_instance, exclude_unset=True)
        logger.info("parsed raw data:")
        logger.info(jdumps(parsed_raw_data))
        self.assertEqual(raw_data, parsed_raw_data)

        with self.assertRaises(Exception) as context:
            logger.info("test invalid data")
            del raw_data["text"]
            parse_obj_as(TextClassifyExample, raw_data)
            self.assertTrue('This is broken' in context.exception)
