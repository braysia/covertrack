from setuptools import setup, find_packages

setup(
    name="covertrack",
    version="0.1",
    packages=find_packages(),
    # packages=["covertrack"],
    author='Takamasa Kudo',
    author_email='kudo@stanford.edu',
    license="MIT License",
    entry_points={
        "console_scripts": [
            "covertrack=covertrack.caller:main",
        ],
    }
)



