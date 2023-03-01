from setuptools import setup, find_packages
from os import getcwd

current_path = getcwd()

# with open("README.md", "r") as fh:
#     long_description = fh.read()

long_description = ""

with open("_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "pandas>=1.3.5",
    "sympy>=1.10.1",
    "numpy>=1.22.3",
    "networkx>=2.8",
    "matplotlib>=3.4.3",
    "scipy>=1.8",
    "docplex>=2.23.1",
    "autograd>=1.4",
    "semantic_version>=2.10",
    "autoray>=0.3.1",
]

requirements_docs = [
    "sphinx>=4.5.0",
    "sphinx-autodoc-typehints>=1.18.1",
    "sphinx-rtd-theme>=1.0.0",
    "nbsphinx>=0.8.9",
    "ipython>=8.10.0",
    "nbconvert>=6.5.1",
]

requirements_test = [
    "pytest>=7.1.0",
    "pytest-cov>=3.0.0",
    "ipython>=8.2.0",
    "nbconvert>=6.5.1",
    "pandas>=1.4.3",
    "plotly>=5.9.0",
    "cplex>=22.1.0.0",
]

package_names = [
    "openqaoa"
]
folder_names = [
    "openqaoa-core"
]
packages_import = find_packages(where="openqaoa-core")
updated_packages = []
for each_package_name in packages_import:
    for _index, each_folder_name in enumerate(folder_names):
        if each_folder_name in each_package_name:
            updated_packages.append(
                each_package_name.replace(each_folder_name, package_names[_index])
            )
            continue

setup(
    name="openqaoa-core",
    python_requires=">=3.8, <3.11",
    version=version,
    author="Entropica Labs",
    packages=updated_packages,
    package_dir={"openqaoa": "openqaoa-core"},
    url="https://github.com/entropicalabs/openqaoa",
    install_requires=requirements,
    license="MIT",
    description="OpenQAOA is a python open-source multi-backend Software Development Kit to create, customise and execute the Quantum Approximate Optimisation Algorithm (QAOA) on Noisy Intermediate-Scale Quantum (NISQ) devices, and simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="quantum optimisation SDK", 
    extras_require={
        "docs": requirements_docs,
        "tests": requirements_test,
        "all": requirements_docs + requirements_test,
    },
)
