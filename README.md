# GradGraph

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/Fricker95/gradgraph/actions/workflows/tests.yml/badge.svg)](https://github.com/Fricker95/gradgraph/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/gradgraph/badge/?version=latest)](https://gradgraph.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/gradgraph.svg?logo=pypi&logoColor=white)](https://pypi.org/project/gradgraph/)
[![PyPI downloads](https://img.shields.io/pypi/dm/gradgraph.svg)](https://pypi.org/project/gradgraph/)

**GradGraph** is a Python library for **gradient-based parameter optimization on graph-structured data**.  
It provides tools to preprocess temporal graphs into optimization-ready arrays and TensorFlow templates for simulating ODE/PDE-based dynamical systems.

---

## Features

- **Graph preprocessing**
  - Extract linear paths from `networkx` graphs
  - Apply sliding-window transformations to generate equal-length arrays
  - Handle temporal node attributes (e.g. position, time of appearance)

- **Differentiable dynamical models**
  - Define custom ODE/PDE-inspired layers (`BasePDESystemLayer`)
  - Use local vs. global trainable parameters with constraints
  - Built on `tf.keras` for automatic differentiation and training

- **Training utilities**
  - `BasePDESystemTrainer` integrates with the full Keras training API
  - Gradient accumulation and optimizer splitting (global vs. local)
  - Early stopping, learning rate scheduling, and checkpointing callbacks

- **Applications**
  - Modeling biological growth (e.g. fungal networks)
  - Simulating transport, epidemiological spread, and other temporal graph processes
  - General gradient-based parameter estimation on irregular domains

---

## Installation

### From PyPI (recommended)

The package is published on [PyPI](https://pypi.org/project/gradgraph/), so you can install it directly with:

```bash
pip install gradgraph
```

### From source

```bash
git clone https://github.com/Fricker95/gradgraph.git
cd gradgraph
pip install -e .
```

Dependencies include: `numpy`, `scipy`, `networkx`, `tensorflow>=2.0`, `matplotlib`.

---

## Examples

See the notebooks in [`examples/`](examples/) (e.g., `example_linear_growth_graph.ipynb`) for end-to-end workflows: loading graphs, feature extraction, interpolation, model setup, training, and evaluation.

---

## Documentation

Full API docs and guides are available on **ReadTheDocs**:  
<https://gradgraph.readthedocs.io/en/latest/>

If you find something missing or unclear, please open an issue or PR.

---

## Community

**Who is this for?**  
Researchers and practitioners in **network science, computational biology, applied mathematics, and machine learning** who fit **dynamical models on graphs** and need efficient, gradient-based parameter estimation.

**How to contribute**

- Open an issue for bugs or feature requests: <https://github.com/Fricker95/gradgraph/issues>  
- Submit a pull request for improvements (tests appreciated!).  
- Follow a clear commit message style and include concise docstrings/type hints where possible.  

If you plan a larger contribution, feel free to start a discussion in an issue to align on design.

---

## License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## Citation

This work was supported by EUR SPECTRUM at Université Côte d’Azur (50%) and by the French National Research Agency (ANR) through the project NEMATIC (50%), grant number ANR-21ANR08Z6RCHX.

If you use **GradGraph** in your research, please cite the associated JOSS paper (pending):

```bibtex
@article{Fricker2025gradgraph,
  title   = {GradGraph: Gradient-based Parameter Optimization on Graph-Structured Data in Python},
  author  = {Fricker, Nicolas E. and Monasse, Laurent and Guerrier, Claire},
  journal = {Journal of Open Source Software},
  year    = {2025},
  note    = {In review},
}
