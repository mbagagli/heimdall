from pathlib import Path
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

# load the version
about = {}
version_file = Path(__file__).parent / "heimdall" / "__init__.py"
exec(version_file.read_text(), about)  # fills the 'about' dict
version = about["__version__"]

setup(
    name="heimdall",
    version=version,
    author="Matteo Bagagli",
    author_email="matteo.bagagli@dst.unipi.it",
    description="a grapH based sEIsMic Detector And Locator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbagagli/heimdall",
    python_requires='>=3.9',
    install_requires=required_list,
    setup_requires=['wheel'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    zip_safe=False,
    scripts=[
        'bin/core/HeimdallCore_1_BuildNetwork.py',
        'bin/core/HeimdallCore_1a_BuildGrid.py',
        'bin/core/HeimdallCore_2_PrepareDataset.py',
        'bin/core/HeimdallCore_2a_createHdf5.py',
        'bin/core/HeimdallCore_3_Training.py',
        'bin/core/HeimdallCore_4_Predict.py',
        'bin/core/HeimdallCore_5_ExtractResults.py',
        #
        'bin/utils/HeimdallUtils_Plot_Event.py',
        'bin/utils/HeimdallUtils_Plot_Catalog.py',
        'bin/utils/HeimdallUtils_Plot_Label.py']
)
