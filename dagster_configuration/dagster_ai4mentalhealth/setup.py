from setuptools import find_packages, setup

setup(
    name="dagster_ai4mentalhealth",
    packages=find_packages(exclude=["dagster_ai4mentalhealth_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "pandas",
        "langchain-openai",
        "langchain-community",
        "langchain",
        "selenium",
        "snowflake-connector-python"
        "chromadb"


    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
