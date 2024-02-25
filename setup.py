from setuptools import setup

setup(
    name="openlst-sw",
    version='0.1',
    packages=["openlst_tools"],
    install_requires=["pyserial", "numpy", "ipython", "click"]
)
