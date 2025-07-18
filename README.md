# LCNN for SU(3)

This repository extends the Lattice gauge equivariant convolutional neural network (L-CNN) of Favoni et al. to the SU(3) group.

---

## Original Work and License

The original contributions can be found at:

- [**arXiv:2012.12901 – Lattice Gauge Equivariant Convolutional Networks**](https://arxiv.org/abs/2012.12901)
- [**Original GitLab repository**](https://gitlab.com/openpixi/lge-cnn/-/tree/prl_2022?ref_type=heads)

The original codebase is licensed under the MIT License.  
This repository **inherits their license**. See the original repository for further details.

---

## Key Implementations

- Extended the architecture from SU(2) to SU(3)
- Evaluation of SU(3) Wilson loops in two dimensions
- Monte Carlo study of the data generation

---
Note:
The current implementation is not compatible with SU(2), as the initialization routine (specifically the flipping behavior) was adapted for SU(3). Support for SU(2) can be reintroduced with minor modifications.

## Acknowledgements

All credit for the architecture and codebase goes to the authors of the original implementation.  
This work should be regarded solely as an extension to their contributions.
