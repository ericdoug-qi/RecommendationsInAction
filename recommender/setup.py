# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: setup.py.py
   Description : 
   Author : ericdoug
   date：2021/1/27
-------------------------------------------------
   Change Activity:
         2021/1/27: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
from os import path

# third packages
from setuptools import setup, find_packages

# my packages

version = __import__('recommender.__init__').VERSION

# Get the long description from the README file
with open(path.join("recommender", "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="recommender",
    version=version,
    description="Recommender Toolkit",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ericdoug-qi/RecommendationsInAction",
    author="EricDoug",
    author_email="ericdoug_qi@163.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="recommendations recommenders recommender system engine machine learning python spark gpu",
    package_dir={"recommender": "recommender"},
    packages=find_packages(where=".", exclude=["tests"]),
    python_requires=">=3.7, <4",
)
