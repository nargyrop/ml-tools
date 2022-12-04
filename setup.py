from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mltoolkit",
    version="0.1.1",
    description="ML/DL Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nargyrop.com",
    author="Nikos Argyropoulos",
    author_email="n.argiropgeo@gmail.com",
    license="MIT",
    packages=["mltoolkit"],
    package_dir={"mltoolkit": "mltoolkit"},
    python_requires=">=3.8",
    zip_safe=False,
    install_requires=[
        "numpy==1.23.5",
        "tensorflow==2.11.0",
        "xmltodict==0.13.0"
    ]
)
