import setuptools

setuptools.setup(
    name="hyperseg",
    version="1.0",
    author="Dr. Yuval Nirkin",
    author_email="yuval.nirkin@gmail.com",
    description="HyperSeg: Patch-wise Hypernetwork for Real-time Semantic Segmentation",
    long_description_content_type="text/markdown",
    package_data={'': ['license.txt']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
