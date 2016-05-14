from setuptools import setup

setup(
    name='luchador',
    version='0.1.0',
    packages=[
        'luchador',
        'luchador.agent',
    ],
    entry_points={
        'console_scripts': [
            'exercise = luchador.exercise:entry_point',
        ]
    },
    test_suite='tests',
    install_requires=[
        'gym',
        'pyyaml',
    ],
    package_data={
        'luchador': ['data/*'],
    },
)
