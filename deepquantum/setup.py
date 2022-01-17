# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:26:20 2022

@author: shish
"""


'''
该脚本用于把deepquantum相关.py文件封装成.pyd文件以加快调用
首先要在powershell或者cmd命令行切换到该脚本所在路径
python setup.py build_ext
建议先安装最新版visual studio并安装C++相关组件，否则可能提示找不到一些文件
'''
#from distutils.core import setup
from setuptools import setup, Extension
from Cython.Build import cythonize

#language_level=3是指定python版本，到底是python3还是python2
setup(
    ext_modules=cythonize("./gates/qTN_contract.py", language_level=3)
)

# setup(
#     ext_modules=cythonize("./embeddings/qembedding.py")
# )
# setup(
#     ext_modules=cythonize("./embeddings/qstate.py")
# )




# setup(
#     ext_modules=cythonize("./gates/qmath.py")
# )
# setup(
#     ext_modules=cythonize("./gates/qcircuit.py")
# )
# setup(
#     ext_modules=cythonize("./gates/qgate.py")
# )
# setup(
#     ext_modules=cythonize("./gates/qoperator.py")
# )
# setup(
#     ext_modules=cythonize("./gates/qtensornetwork.py")
# )




# setup(
#     ext_modules=cythonize("./layers/qlayer.py")
# )
# setup(
#     ext_modules=cythonize("./layers/qlayers.py")
# )