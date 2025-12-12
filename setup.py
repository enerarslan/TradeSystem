"""
AlphaTrade System Setup Configuration
JPMorgan-Level Institutional Trading Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="alphatrade",
    version="1.0.0",
    author="AlphaTrade Team",
    author_email="team@alphatrade.dev",
    description="JPMorgan-Level Institutional Algorithmic Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alphatrade/alphatrade-system",
    project_urls={
        "Bug Tracker": "https://github.com/alphatrade/alphatrade-system/issues",
        "Documentation": "https://alphatrade.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphatrade=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
