from setuptools import setup, find_packages

setup(
    name="upsaclay_ml",
    version="1.0.0",
    description="A Machine Learning Library Built with OOP Principles learning the hard way",
    author="group9 M1 Geniomhe - Univ. Évry Paris-Saclay",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "requests>=2.0"
    ],
    python_requires=">=3.8",
)
