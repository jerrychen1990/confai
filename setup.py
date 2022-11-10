#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: setup.py
@time: 2022/11/10 14:17
"""

import os
import re
import subprocess

from setuptools import setup, find_packages


# 获取某个包的最新版本
def get_last_version(package, index_url="https://pypi.org/simple"):
    cmd = f"pip index versions {package}"
    if index_url:
        cmd = f"{cmd} --index-url {index_url}"
    print(cmd)
    status, output = subprocess.getstatusoutput(cmd)

    pattern = re.compile("Available versions:(.*)")
    match = list(re.findall(pattern, output))
    # print(match)
    if not match:
        print(f"no available version found by output:\n{output}")
        return None
    versions = sorted([tuple(int(v) for v in e.strip().split(".")) for e in match[0].split(",")])
    return versions[-1]


# 获取下一个版本
def get_next_version(base_version, last_version):
    print(f"base_version: {base_version}, last_version: {last_version}")
    if last_version:
        last_base = tuple(last_version[:2])
        if base_version and last_base < base_version[:2]:
            return base_version
        else:
            return last_version[:2] + (last_version[2] + 1,)

    return base_version


def get_base_version(pkg_name):
    try:
        libinfo_py = os.path.join(pkg_name, '__init__.py')
        libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
        version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
            0
        ]
        exec(version_line)
        return tuple([int(e) for e in locals()["__version__"].split(".")])

    except FileNotFoundError:
        return None


INSTALL_REQUIRES = [
    "requests",
    "numpy",
    "transformers",
    "tqdm",
    "tritonclient[http]",
    "whale-ai-schema",
    "pandas",
    "datasets",
    "click",
    "torch",
    "accelerate",
    "onnx"
]
DEPENDENCY_LINKS = [
    "https://download.pytorch.org/whl/cu113"
]

if __name__ == "__main__":
    name = "confai"
    version = get_next_version(get_base_version(name), get_last_version(name))
    version = ".".join([str(e) for e in version])
    print(f"version: {version}")
    setup_kwargs = dict(
        name=name,
        packages=find_packages(),
        version=version,
        include_package_data=True,
        description="build ai with config",
        author='Chen Hao',
        author_email='jerrychen1990@gmail.com',
        license='MIT',
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        dependency_links=DEPENDENCY_LINKS,
        setup_requires=['setuptools>=18.0', 'wheel'],
        keywords='furnace ai',
    )

    setup(**setup_kwargs)
