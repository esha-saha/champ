# Constrained Hybrid Approaches for Methane Prediction
Code implementations and repository of the paper titled 'Methane projections from Canada's oil sands tailings using scientific deep learning reveal significant underestimation' (https://arxiv.org/abs/2411.06741) currently under review.

## Installation

Use the steps below for a local installation of the model and to reproduce results in the paper.

```
$ git clone https://github.com/esaha2703/champ.git
$ pip install - e .
```
## Problem Statement

The goal of the proposed research is to train a parameterized model to track emissions from oil sands tailings ponds in Alberta, Canada using methane concentrations.

For each $i^{th}$ observations in the set $\mathcal{I}_{obs}$, suppose  $\mathbf{x}_i$ denotes the $(d+1)-$ dimensional input vector and $u_i$ denotes observed output. Then given a fixed function $q:\mathbb{R}^{d+1}\rightarrow\mathbb{R}$ describing emission dynamics (from MMs), find a function $u:\mathbb{R}^{d+1}\rightarrow \mathbb{R}$ by solving the problem,

$\min_{\phi}$ $\frac{1}{|\mathcal{I}_{obs}|}$ $\sum(u( \mathbf{x}_i ) - u_i )^2$ subject to  $F(\phi(\mathbf{x},u),u,q)=0$

where $F$ is the physical constraint with unknown function $\phi$. 

Our choice of the constraint $F$ is derived from an atmospheric dispersion models called Gaussian Plume Model (GPM), 

$\dfrac{\partial u}{\partial t} + \nabla.J = q $

where $u(\vec{x},t)$ is the mass concentration, $q(\vec{x},t)$ is a source (or sink) and $J$ is mass flux due to diffusion ($J_D$) and advection ($J_A$), and $\vec{x}$ and $t$ denote space and time respectively. 

## Implementations

1. CHAMP_EvaluationFinal.ipynb for evaluating the trained models on training and test data. Use the uploaded trained models for evaluation
2. CHAMP_Training.ipynb for training models from scratch
3. CHAMP_helpers.py contains helper functions for the above notebooks

## Citation and Contact
If you use any part of this work or find it useful in any way, please cite. For any comments or questions please email esaha1@ualberta.ca
```
@article{saha2024methane,
  title={Methane projections from Canada's oil sands tailings using scientific deep learning reveal significant underestimation},
  author={Saha, Esha and Wang, Oscar and Chakraborty, Amit K and Garcia, Pablo Venegas and Milne, Russell and Wang, Hao},
  journal={arXiv preprint arXiv:2411.06741},
  year={2024}
}
```
