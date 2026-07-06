from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]


setup(
    name='diffsci',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    version='0.1.0',
    description='Diffusion models for scientific applications',
    author='UFRJ',
    license='BSD-3',
)
