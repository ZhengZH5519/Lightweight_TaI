from setuptools import setup, find_packages

setup(
    name="Lightweight_TaI",
    version="0.0.1",
    description="Lightweight tools: TopK compression and EngineBuilder for MTQ -> TensorRT",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="ZhengZH5519",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pillow",
    ],
    zip_safe=False,
)
