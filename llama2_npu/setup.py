from setuptools import find_packages, setup
import os
DEVICE_ID= 0
if os.getenv('DEVICE_ID') and str.isdigit(os.getenv('DEVICE_ID')):
    DEVICE_ID= int(os.getenv('DEVICE_ID'))


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def fetch_version():
    with open("version.txt", "r") as f:
        return f.read().strip()


setup(
    name="colossal_llama",
    version=fetch_version(),
    packages=find_packages(exclude=("*.egg-info",)),
    description="Continual Pre-training and SFT for LLaMA",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA",
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
