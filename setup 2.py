from setuptools import setup, find_packages

setup(
    name="turbofan-predictor",
    version="0.1.0",
    description="Predictive maintenance for aircraft turbofan engines",
    author="Nia Racheva",
    author_email="niaracheva05@gmail.com",
    url="https://github.com/nia05-tch/aircraft-engine-predictive-maintenance",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
)
