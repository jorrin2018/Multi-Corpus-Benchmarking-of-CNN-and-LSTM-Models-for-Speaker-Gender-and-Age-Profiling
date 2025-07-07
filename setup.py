"""
Setup configuration for Speaker Profiling Benchmark package.

This package provides a comprehensive benchmarking suite for speaker profiling 
models (gender and age classification) across VoxCeleb1, Common Voice, and TIMIT datasets.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for package description."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Package metadata
__version__ = "1.0.0"
__author__ = "Jorge Jorrin-Coz et al."
__email__ = "jljorrincoz@gmail.com"
__description__ = "Multi-Corpus Speaker Profiling Benchmark: CNN & LSTM Models for Gender and Age Classification"

setup(
    # Basic package information
    name="speaker-profiling-benchmark",
    version=__version__,
    description=__description__,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # Author information
    author=__author__,
    author_email=__email__,
    
    # URLs
    url="https://github.com/your-username/speaker-profiling-benchmark",
    project_urls={
        "Bug Reports": "https://github.com/your-username/speaker-profiling-benchmark/issues",
        "Source": "https://github.com/your-username/speaker-profiling-benchmark",
        "Documentation": "https://github.com/your-username/speaker-profiling-benchmark/blob/main/README.md",
    },
    
    # License
    license="MIT",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements("requirements.txt"),
    
    # Optional dependencies for development
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "testing": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
        ],
        "mlops": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "mlflow>=1.20.0",
        ]
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "config": ["*.yaml", "*.yml"],
    },
    
    # Scripts and entry points
    entry_points={
        "console_scripts": [
            "sp-train=scripts.train:main",
            "sp-evaluate=scripts.evaluate:main", 
            "sp-predict=scripts.predict:main",
            "sp-benchmark=scripts.benchmark:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords for discoverability
    keywords=[
        "speaker profiling", "gender classification", "age classification",
        "audio processing", "deep learning", "CNN", "LSTM", "VoxCeleb1", 
        "Common Voice", "TIMIT", "benchmarking", "reproducible research"
    ],
    
    # Minimum versions for key dependencies
    zip_safe=False,  # Package contains data files
    
    # Additional metadata
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=6.2.0",
        "pytest-cov>=3.0.0",
    ],
) 