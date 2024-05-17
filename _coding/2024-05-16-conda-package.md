---
layout: post
title: "Creating a conda package"
katex: False
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

STEP 1: Install the `anaconda-client` and `conda-build` into your virtual environment.

```bash
(venv)$ conda install anaconda-client conda-build
```
Suppose your directory structure looks like the following.

```bash
(venv)$ tree mypackage

mypackage
├── README.md
├── example
│   └── test.ipynb
├── setup.py
└── mypackage
    ├── __init__.py
    └── myfile.py
```
STEP 2: To package this code into a conda package make a directory in the root repository `mkdir conda-recipe`. Add a file to this directory called `meta.yaml`. Your directory structure should now look like

```bash
(venv)$ tree mypackage

mypackage
├── README.md
├── conda-recipe
│   └── meta.yaml
├── example
│   └── test.ipynb
├── setup.py
└── mypackage
    ├── __init__.py
    └── myfile.py
```

Here is an example of what a `meta.yaml` file looks like. 

```yaml
# meta.yaml file
package:
  name: mypackage
  version: 0.1.0

source:
  git_url: https://github.com/path/to/repo.git
  git_rev: main # or some other branch of the repo
# alternatively, you can build from local directory using
# path: ../

build:
  script: {% raw %} {{ <span style="color:#d44950;">PYTHON</span> }} {% endraw %} -m pip install . -vv

requirements:
  build:
    - pip
    - setuptools
    - python
    - git
  run:
    - numpy
    - scipy
```

The `build` and `run` requirements indicate the packages required for building and running the package, respectively. The `script` tells us the command that will be run to build the package.

STEP 3: Navigate to the root directory and run:
```bash
(venv)$ conda build .
```
If successful, the console should print information about the location of the compressed package ready for upload. 

STEP 4: To upload it to your anaconda channel, run:
```bash
anaconda upload /path/to/the/package/mypackage-0.1.0-0.tar.bz2
```

STEP 5: To install this package, run:
```bash
(venv)$ conda install my-channel-name::mypackage
```



