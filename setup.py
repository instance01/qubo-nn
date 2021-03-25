from setuptools import setup

setup(
    name='qubo-nn',
    version='0.1.0',
    install_requires=[
        'numpy', 'networkx', 'torchvision', 'torch', 'dwave-qbsolv'
    ]
)
