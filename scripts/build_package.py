#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: build_package.py
@time: 2022/11/10 17:05
"""

import click
import subprocess


def exec_cmd(cmd):
    print("*" * 30)
    print(f"EXECUTE: {cmd}")
    status, output = subprocess.getstatusoutput(cmd)
    print(f"cmd ends with\n status:{status}\n output:\n{output}")
    print("EXECUTE DONE")
    print("*" * 30)
    if status != 0:
        raise Exception(f"cmd:{cmd} failed with status:{status}")


@click.command()
@click.option("--upload/--no-upload", default=False, help="是否要上传到pypi.meetwhale.com")
def build_package(upload):
    print("removing files...")
    exec_cmd("rm -rf dist build *.egg-info")
    print("packaging...")
    exec_cmd(f"python setup.py sdist  bdist_wheel")
    if upload:
        print("uploading...")
        exec_cmd("python -m twine upload -r pypi --verbose dist/*.whl")
        print("removing files...")
        exec_cmd("rm -rf dist build *.egg-info")
    print("job done")


if __name__ == "__main__":
    build_package()
