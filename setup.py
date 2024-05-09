from setuptools import setup, find_packages


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


requirements = parse_requirements("requirements.txt")

setup_requirements = [
    "pytest-runner",
]

test_requirements = parse_requirements("requirements_dev.txt")

setup(
    author="Caio de Souza Barbosa Costa",
    author_email="csbc326@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    description="Reinforcement learning framework to solve portfolio optimization tasks.",
    install_requires=requirements,
    license="MIT License v3",
    # long_description=readme + '\n\n' + history, # TO-DO
    include_package_data=True,
    keywords="rl_portfolio",
    name="rl_portfolio",
    packages=find_packages(include=["rl_portfolio.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/C4i0kun/RLPortfolio",
    version="0.0.1",
    zip_safe=False,
)
