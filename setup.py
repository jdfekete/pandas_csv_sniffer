#!/usr/bin/env python

from setuptools import setup

setup(
    name='pandas_csv_sniffer',
    version='0.1',
    description='Jupyter notebook GUI to pandas.csv_read',
    author='Jean-Daniel Fekete',
    author_email='Jean-Daniel.Fekete@inria.fr',
    url='https://github.com/jdfekete/pandas_csv_sniffer',
    license="MIT",
    packages=['csv_sniffer'],
    platforms='any',
    setup_requires=['nose'],
    test_suite='nose.collector',
    install_requires=[
        "pandas",
        "fsspec",
        "ipywidgets"
    ]
)
