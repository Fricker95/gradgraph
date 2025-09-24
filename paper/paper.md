---
title: 'GradGraph: Gradient-based Parameter Optimization on Graph-Structured Data in Python'
tags:
  - Python
  - Graphs
  - Gradient Optimization
  - Gradient Descent
  - Machine Learning
  - Network Science
  - Data-driven Modeling
  - Dynamical Systems
  - PDEs
  - ODEs
  - Tensorflow
authors:
  - name: Nicolas E. Fricker
    orcid: 0009-0002-9027-4111
    corresponding: true
    equal-contrib: false
    affiliation: 2
  - name: Laurent Monasse
    orcid: 0000-0001-8090-9107
    equal-contrib: false
    affiliation: 3
  - name: Claire Guerrier
    orcid: 0000-0002-1128-1069
    equal-contrib: false
    affiliation: "1, 2"
affiliations:
 - name: Centre de Recherches Mathématiques (CRM), Université de Montréal, CNRS, Montréal, Canada
   index: 1
 - name: Université Côte d’Azur, CNRS, Laboratoire J. A. Dieudonné (LJAD, UMR 7351), Nice, France
   index: 2
 - name: Université Côte d’Azur, Inria, CNRS, Laboratoire J. A. Dieudonné (LJAD, UMR 7351), EPC ACUMES, Nice, France
   index: 3
date: 31 August 2025
bibliography: paper.bib
---

# Summary

Many scientific systems can be modeled as **dynamical processes on graphs**, such as the spread of disease in populations, flows in infrastructure networks, or transport phenomena across irregular domains [@Enright2018; @Neri2013]. To fit such models to observed data, researchers need both (1) a way to transform irregular graph-structured data into optimization-ready arrays, and (2) a framework for efficient parameter estimation using gradient descent. This work was motivated by challenges we encountered when modeling fungal growth, which required efficient parameter optimization in a graph-based framework.

**GradGraph** provides this functionality. It preprocesses graphs by extracting **linear paths** and applying a **moving-window (overlapping span) approach** to generate standardized arrays of equal length. This converts irregular, graph-based observations into a structured dataset suitable for machine learning pipelines. On top of this preprocessing, GradGraph supplies **TensorFlow templates** for simulating systems of ODEs or PDEs on these arrays and optimizing parameters via gradient-based methods.  

# Statement of need

While frameworks like `TensorFlow` [@tensorflow] and `PyTorch` [@pytorch] provide robust automatic differentiation, they do not directly support the **graph-to-array preprocessing pipeline** required to model dynamical systems on networks. Users must typically:  

- Manually traverse graphs to extract data sequences,  
- Implement custom windowing schemes to standardize sequence lengths,  
- Build training loops for ODE/PDE parameter optimization.  

`GradGraph` addresses these challenges by:  

- Extracting **linear paths** from graphs using `networkx` and related tools,  
- Applying a **sliding-window transformation** to produce many equal-length arrays from each path,  
- Providing **ready-to-use TensorFlow templates** for ODE/PDE models,  
- Enabling **gradient-based optimization** of model parameters with minimal boilerplate,  
- Remaining compatible with the broader Python scientific stack (`numpy`, `scipy`, `networkx`).  

This combination makes GradGraph a powerful and accessible framework for researchers and practitioners who want to calibrate dynamical models on graph-structured data.  

# Framework

Let $G = (V, E)$ be a graph with data $D: V \to \mathbb{R}^m$. For a linear path  
$$P = (v_1, v_2, \dots, v_L) \subseteq V,$$  
GradGraph constructs overlapping windows of fixed length $w$,  

$$
W_j = (v_j, v_{j+1}, \dots, v_{j+w-1}), \quad j = 1, \dots, L-w+1.
$$  

From each window $W_j$, an array
$$X_{W_j} \in \mathbb{R}^{w \times m}$$
is created. This ensures all arrays have the same shape, making them suitable for TensorFlow-based optimization.  

For a model $M_\theta$ (ODE or PDE system) parameterized by $\theta$, GradGraph defines a loss  

$$
\mathcal{L}(\theta) = \sum_{W} \mathcal{L}_W\!\left(M_\theta(X_W), D_W\right),
$$  

where $D_W$ are the observed data restricted to window $W$. TensorFlow’s autodiff then computes  

$$
\nabla_\theta \mathcal{L},
$$  

enabling optimization via standard methods (SGD, Adam, etc.).

The intended user community for GradGraph includes researchers and practitioners in **network science, computational biology, applied mathematics, and machine learning** who are interested in fitting dynamical models to graph-structured data.  
Typical applications include the study of **biological growth processes, epidemiological spread, transport phenomena, and infrastructure networks**, where processes evolve on irregular domains and efficient **gradient-based parameter estimation** is required.

# Citations

GradGraph builds on the Python ecosystem: `networkx` [@networkx] for graph handling, `numpy` [@numpy] and `scipy` [@scipy] for numerical routines, and TensorFlow [@tensorflow] for autodiff and optimization.  

# Acknowledgements

This work was supported by EUR SPECTRUM at Université Côte d’Azur (50%) and by the French National Research Agency (ANR) through the project NEMATIC (50%), grant number ANR-21ANR08Z6RCHX.

# References
