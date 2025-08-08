"""
Setup script for the Agentic AI Forecasting System
"""
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentic-forecasting-system",
    version="1.0.0",
    author="Codegen AI",
    author_email="support@codegen.com",
    description="A sophisticated multi-agent AI system for time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ritwikv/Forecasting_Agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Spreadsheet",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "llama-cpp-python[cublas]",  # GPU support
        ],
    },
    entry_points={
        "console_scripts": [
            "forecasting-system=run_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="forecasting, time-series, ai, agents, langchain, streamlit, phi3",
    project_urls={
        "Bug Reports": "https://github.com/ritwikv/Forecasting_Agents/issues",
        "Source": "https://github.com/ritwikv/Forecasting_Agents",
        "Documentation": "https://github.com/ritwikv/Forecasting_Agents/blob/main/docs/README.md",
    },
)

