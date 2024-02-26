import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "SignLanguageTranslator_using_MLflow_and_DVC"
AUTHOR_USER_NAME = "himanko"
SRC_REPO = "SignLanguageTranslator"
AUTHOR_EMAIL = "himankoboruah@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A attention based Sign Language Translator to detect and translate the isolated charecters. The model is trained under two ISL dataset and a ASL datasrt. ISL = INCLUDE and a small local dataset; ASL = WLASL. For feature extraction I used MediaPipe and saved the landmarks using .perquet and .npy",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)