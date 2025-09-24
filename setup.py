"""專案安裝設定檔"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="surgery-duration-prediction",
    version="1.0.0",
    author="李泓斌",
    author_email="leeyuchen0321@gmail.com",
    description="手術時間預測系統",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/surgery-duration-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-surgery-models=scripts.train_all_departments:main",
        ],
    },
)
