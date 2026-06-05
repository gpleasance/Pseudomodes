from setuptools import setup, find_packages

REQUIRES = ['numpy', 'scipy', 'qutip', 'oqupy']

setup(
    name="pseudomodes",
    packages=find_packages(),
)