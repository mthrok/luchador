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
            'exercise = luchador.exercise:main',
        ]
    },
    test_suite='tests',
    install_requires=[
        'gym',
    ],
)
