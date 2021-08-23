# Coherence-Review-Generation
Coda and Data source for the SIGIR 2021 paper "[Knowledge-based Review Generation by Coherence Enhanced Text Planning](https://dl.acm.org/doi/abs/10.1145/3404835.3462865)"

# Directory

- [Requirements](#Requirements)
- [Datasets](#Datasets)
- [Training Instructions](#Training-Instructions)
- [Testing Instructions](#Testing-Instructions)
- [License](#License)
- [Reference](#References)

# Requirements

- Python 3.7
- Pytorch 1.8
- torch-geometric 
- Anaconda3

# Datasets

Our datasets are the same as our paper "[Knowledge-Enhanced Personalized Review Generation with Capsule Graph Neural Network](https://arxiv.org/abs/2010.01480)"(CIKM 2020). You can refer to [this reppository](https://github.com/turboLJY/CapsGNN-Review-Generation).

# License

```
License agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions. 
2. That you include a reference to the dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the dataset website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China).
```

# References

If this work is useful in your research, please cite our paper.

```
@inproceedings{junyi2021planning,
  title={{K}nowledge-based {R}eview {G}eneration by {C}oherence {E}nhanced {T}ext {P}lanning},
  author={Junyi Li, Wayne Xin Zhao, Zhicheng Wei, Nicholas Jing Yuan, Ji-Rong Wen},
  booktitle={SIGIR},
  year={2021}
}
```

