#!/usr/bin/env python
# encoding: utf-8
"""
@author: chenhao
@file: test.py
@time: 2022/11/23 14:56
"""
from typing import Optional, Union

from pydantic import BaseModel


class A(BaseModel):
    name: str


class B(A):
    age: int


class C(A):
    gender: str


Example = Union[C, B, A]


# if __name__ == '__main__':
#     app.run(host="
class Examples(BaseModel):
    data: list[Union[A, B, C]]


def build_union_instance(union_cls, data):
    for cls in union_cls.__args__:
        try:
            print(f"cls:{cls}, data:{data}")
            rs = cls(**data)
            return rs
        except Exception as e:
            # logging.exception(e)
            pass
    raise Exception("build union instance failed")


tmp = {"name": "c", "age": 10}
rs = build_union_instance(Example, tmp)
print(rs)
print(type(rs))
