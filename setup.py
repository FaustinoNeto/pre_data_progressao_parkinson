from setuptools import setup, find_packages

setup(
    name="pre_data_progressao_parkinson",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
    ],
    entry_points={
        'console_scripts': [
            'run_parkinson=pre_data_progressao_parkinson.cli:main'
        ]
    },
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    description="Pipeline e modelagem para a progressão da Doença de Parkinson",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
