DiffSci
==============================

Neural networks for porous media

--------

For a tutorial, see notebooks/tutorials/0001-basic-usage.ipynb.


==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks
    │   ├── exploratory    <- Notebooks for exploration and experiments
    │   └── tutorials      <- Tutorial notebooks showing how to use the library
    │
    ├── diffsci            <- Source code for use in this project.
    │   ├── __init__.py    <- Makes diffsci a Python module
    │   ├── models         <- Model implementations
    │   │   ├── karras     <- Karras et al. diffusion models
    │   │   ├── nets       <- Neural network architectures
    │   │   └── trainers   <- Training utilities
    │   └── utils          <- Utility functions
    │
    ├── tests             <- Unit tests
    │
    ├── requirements.txt   <- Project dependencies
    │
    ├── saveddata         <- Data directory
    │   ├── external      <- Data from third party sources
    │   ├── interim       <- Intermediate data
    │   ├── processed     <- Final, processed datasets
    │   └── raw           <- Original, immutable data
    │
    ├── savedmodels      <- Trained models
    ├── saveddata        <- Saved data files
    │
    └── setup.py         <- Makes project pip installable (pip install -e .)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>