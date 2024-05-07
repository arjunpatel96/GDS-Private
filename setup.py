import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lifelike_gds",
    version="0.0.1",
    author="Lifelike Team, Ethan Sanchez <https://github.com/esanche06>, Nathan Weinstein <https://github.com/NathanWeinstein>, Dominik Maszczyk <https://github.com/Skitionek>, Alessandro Negro <https://github.com/alenegro81>",
    author_email="lifelike@biosustain.dtu.dk",
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
    install_requires=[
        "pandas~=1.3.3",
        "numpy~=1.21.2",
        "simplejson~=3.17.5",
        "networkx~=2.6.3",
        "scipy~=1.7.1",
        "python-arango~=7.5.4",
        "xlsxwriter~=3.0.1",
    ]
)
