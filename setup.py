from setuptools import setup, find_packages

setup(
    name="pyderivatives",
    version="0.1.0",  # Initial version
    description="A package for pricing derivatives using Black-Scholes, Monte Carlo, LS-MC, and SABR models",
    author="Omar Debbagh",
    author_email="omar_debbagh@outlook.com",
    url="https://github.com/omardebbagh/pyderivatives",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "yfinance>=0.1.70"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  
)
