# Unc-DepthCompletion-Master

This is the reference PyTorch implementation for training and testing depth completion models using the method described in

> **Robust depth completion with uncertainty-driven loss functions**

If you find our work useful in your research please consider citing our paper:

@inproceedings{zhu2022robust,
  title={Robust depth completion with uncertainty-driven loss functions},
  author={Zhu, Yufan and Dong, Weisheng and Li, Leida and Wu, Jinjian and Li, Xin and Shi, Guangming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={3626--3634},
  year={2022}
}

## Training
STEP ONE(MDCnet):
>python3 train_step1.py --data-folder path/to/data -b 4

STEP TWO(AIRnet):
>python3 train_step2.py --data-folder path/to/data -b 4 --MDCnet-path /path/to/checkpoint

## EVALLATION
>python3 eval.py --MDC-path path/to/checkpoint --AIR-path path/to/checkpoint

