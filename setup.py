#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(name='cvlization',
      version='0.1.3',
      description='Training pipelines for Computer Vision',
      author='KUNGFU.AI',
      author_email='zz@kungfu.ai',
      url='https://github.com/kungfuai/cvlization',
      license='MIT',
      # packages=['cvlization'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
    )
