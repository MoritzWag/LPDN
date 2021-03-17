import setuptools

setuptools.setup(
    name="lpdn",
    version="0.0.2",
    author="Moritz Wagner",
    author_email="moritzwagner95@hotmail.de",
    description="Implementation of Lightweight Probabilistic Deep Network (inference-only) PyTorch",
    url="https://github.com/MoritzWag/LPDN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
