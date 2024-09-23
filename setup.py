from setuptools import find_packages, setup

setup(
    name='diffsci',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'einops',
        'tqdm',
        'safetensors',
        'porespy',
        'numpy',
        'matplotlib',
        'diffusers',
        'lightning',
        'transformers',
        'netCDF4',
        'jaxtyping'
    ],
    version='0.1.0',
    description='Diffusion models for scientific applications',
    author='UFRJ',
    license='BSD-3',
)
