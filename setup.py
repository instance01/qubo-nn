from setuptools import setup

setup(
    name='qubo-nn',
    version='0.2.3',
    install_requires=[
        'numpy', 'networkx', 'torchvision', 'torch', 'dwave-qbsolv',
        'qubovert', 'matplotlib', 'scipy', 'tensorflow', 'sklearn',
        'tensorboard',
        'ml-pyxis@git+https://github.com/vicolab/ml-pyxis@master',
    ],
    long_description_content_type='text/markdown',
    description='QUBO translations for 14 problems. Also: Reverse-engeering and AutoEncoders for QUBOs.',
    url='https://github.com/instance01/qubo-nn',
    author='Instance01',
    author_email='whodis@instancedev.com',
    license='MIT',
    packages=['qubo_nn'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
    ],
    long_description=open('README.md').read(),
    zip_safe=False)
