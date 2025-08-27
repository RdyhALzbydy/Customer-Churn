from setuptools import setup, find_packages

setup(
    name="radiya",
    version="1.0.0",
    description="Customer Churn Prediction for Music Streaming Service",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
)