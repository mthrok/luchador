from setuptools import setup

setup(
    name='luchador',
    version='0.1.0',
    packages=[
        'luchador',
        'luchador.nn',
        'luchador.nn.core',
        'luchador.nn.core.base',
        'luchador.nn.core.theano',
        'luchador.nn.core.tensorflow',
        'luchador.agent',
        'luchador.img_proc',
    ],
    entry_points={
        'console_scripts': [
            'luchador = luchador.exercise:entry_point',
        ]
    },
    test_suite='tests',
    install_requires=[
        'gym',
        'h5py',
        'Pillow',
        'pyyaml',
    ],
    package_data={
        'luchador': [
            'data/*.yml',
            'nn/data/*.yml',
        ],
    },
)
