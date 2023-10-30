import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lifelike_gds",
    version="0.0.1",
    author="Ethan Sanchez",
    author_email="ethan.dsanch@gmail.com",
    description="A collection of GDS packages used by the Lifelike application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SBRG/GDS",
    project_urls={
        "Bug Tracker": "https://github.com/SBRG/GDS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)
