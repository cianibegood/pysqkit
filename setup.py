from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

setup(
    name="pysqkit",
    version="0.0.1",
    description="Python superconducting qubit circuit analysis package",
    long_description=README,
    author="Alessandro Ciani",
    author_email="alessandrociani89@gmail.com",
    url="https://github.com/cianibegood/pysqkit",
    license=LICENSE,
    packages=find_packages("."),
    ext_package="pysqkit",
    install_requires=list(open("requirements.txt").read().strip().split("\n")),
)
