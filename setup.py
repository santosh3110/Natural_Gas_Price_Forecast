import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("-e")
    ]

__version__ = "0.1.0"

REPO_NAME = "Natural_Gas_Price_Forecast"
AUTHOR_USER_NAME = "santosh3110"
SRC_REPO = "gaspriceforecast"
AUTHOR_EMAIL = "santoshkumarguntupalli@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Python package for Natural Gas Price Forecasting using LSTM and time series ML techniques",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires='>=3.8',
    install_requires=requirements,
    include_package_data=True
)
