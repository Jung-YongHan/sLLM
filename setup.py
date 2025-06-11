from setuptools import find_packages, setup

setup(
    name="sLLM",
    version="0.1.0",
    packages=find_packages(include=["sLLM", "sLLM.*"]),
)
