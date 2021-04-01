from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pysqkit',
    version='0.0.1',
    description='Python superconducting qubit circuit analysis package',
    long_description=readme,
    author='Alessandro Ciani',
    author_email='a.ciani@tudelft.nl',
    url='https://github.com/cianibegood/pysqkit',
    license=license,
    packages=find_packages('.'),
    ext_package='pysqkit',
    install_requires=list(open('requirements.txt').read().strip().split('\n'))
)
