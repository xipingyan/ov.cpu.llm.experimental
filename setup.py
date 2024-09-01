import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# execute following command to build w/o install
#    python setup.py build

__version__ = "0.1"

ext_modules = [
    Pybind11Extension("ovllm.utils_cpp",
        ["ovllm/cpp/main.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        ),
]



setuptools.setup(
    name='ovllm',
    version="0.1",
    packages=["ovllm"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    setup_requires=["pybind11"]
)
