from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "example",
        ["cpp/example.cpp"],
        extra_compile_args=["-O3"],
    ),
]

# setup config
setup(
    name="example",
    version="0.1",
    author="xavier",
    description="A simple Pybind11 example",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)