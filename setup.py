from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    # Basic package information:
    name="photonion",
    version="0.1",
    packages=find_packages(),

    # Package metadata:
    author="Nicholas Choustikov & Richard Stiskalek",
    author_email="nicholas.choustikov@physics.ox.ac.uk",
    description="Ionizing Luminosities with ILI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chousti/photometry-nion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy",
                      "matplotlib",
                      "ipympl",
                      "scikit-learn",
                      "joblib",
                      "optuna",
                      "pandas",
                      "tqdm",
                      "h5py",
                      "astropy",
                      ],
)
