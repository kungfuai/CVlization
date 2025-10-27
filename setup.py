#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name="cvlization",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Training and inference recipes for Vision, Language and more modalities.",
    author="KUNGFU.AI",
    author_email="zz@kungfu.ai",
    url="https://github.com/kungfuai/cvlization",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
