"""
Setup script for installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vizon-training",
    version="1.0.0",
    author="Eureka Nitze",
    author_email="eureka@vveesee.com",
    description="YOLO Object Detection Training Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/entz77/vizon_training",
    packages=find_packages(),
    package_data={
        "src.data": ["*.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.3.0",
        "rfdetr[train]>=1.11.2",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
)
