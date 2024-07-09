# BERTPhylo: An Efficient Method to accelerate Phylogenetic Inference using a Pretrained DNA Language Model

[![Project](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/danruod/BERTPhylo)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/danruod/BERTPhylo/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/danruod/BERTPhylo/blob/main/DATA_LICENSE)


> **Authors**: Danruo Deng, Wuqin Xu*, Bian Wu, Hans Peter Comes, Yu Feng, Pan Li*, Jinfang Zheng*, Guangyong Chen*, and Pheng-Ann Heng.  
 **Affiliations**: CUHK, Zhejiang Lab, Zhejiang University, Salzburg Universit, Chengdu Institute of Biology (CAS).


<!-- Understanding the phylogenetic relationships among species is crucial for comprehending major evolutionary transitions, serving as the foundation for many biological studies. Despite the ever-growing volume of sequence data providing a significant opportunity for biological research, constructing reliable phylogenetic trees effectively becomes more challenging for current analytical methods. In this study, we introduce a novel solution to accelerate phylogeny inference using a pretrained DNA language model. Our approach identifies the taxonomic unit of a newly collected sequence using existing taxonomic classification systems and updates the corresponding subtree, akin to surgical corrections on a given phylogenetic tree. Specifically, we leverage a pretrained BERT network to obtain high-dimensional sequence representations, which are used not only to determine the subtree to be updated but also identify potentially valuable regions for subtree construction. We demonstrate the effectiveness of our method, named BERTPhylo, through experiments on our established PlantSeqs dataset, focusing on Embryophyta. Our findings provide the first evidence that phylogenetic trees can be constructed by automatically selecting the most informative regions of sequences, without manual selection of molecular markers. This discovery offers a robust guide for further research into the functional aspects of different regions of DNA sequences, enriching our understanding of biology.

<img src="./assets/BERTPhylo.jpg" alt="drawing" width="80%"/> -->

This repository contains the code necessary to reproduce the experiments results of novelty detection and taxonomic classification of BERTPhylo.

## System Requirements

We here discuss hardware and software system requirements.

### Hardware Dependencies

Our experiments require modern computer hardware suitable for machine learning research. We recommend using one or more graphics processor units (GPUs) to accelerate model inference. Without a GPU, it may be difficult to reproduce our results in a reasonable amount of time.


### Software Dependencies

Our code relies on Python 3.11.9 with PyTorch 2.0.1. We list the exact versions for all packages in [environment.yaml](environment.yaml).


## Installation Guide

To install Python with all necessary dependencies, we recommend the use of conda, and we refer to [https://conda.io/](https://conda.io/) for an installation guide. After installing conda, please execute the following commands to download the code and set up a new conda environment with all required packages:


```
git clone --recursive https://github.com/danruod/BERTPhylo.git
  
conda env create -f environment.yml
conda activate bertphylo
```

We recommend running on a Linux system. The setup should be completed within a few minutes. 

The PlantSeqs dataset needs to be [downloaded](https://drive.google.com/drive/folders/1wAQVjLYqlRA_0Xk9I3XvOsh_A-sdb9zZ?usp=sharing) manually and stored in the plantseqs folder under the BERTPhylo directory (`./plantseqs/`).


## Reproduction Instructions

Run

  ```
  python eval.py  --model_name bertphylo --batch_size 64 --suffix test
  ```

to reproduce results for novelty detection and taxonomic classification on the PlantSeqs dataset. After executing the above code, you can get the outputs at `/results/bertphylo`, where `test.xlsx` contains the source data of all tables in the manuscript, `confusion_matrix`, `ood_scores`, and `pr_roc` folders contain the source data required for Fig. 4b, Fig. 5, Fig. S1-S5 in the manuscript.


## File Structure

 * [`model/`](https://github.com/danruod/BERTPhylo/tree/main/model) directory provides model implementation of BERTPhylo. 
 * [`checkpoints/`](https://github.com/danruod/BERTPhylo/tree/main/checkpoints) directory contains the trained BERTPhylo model parameters. 
 * [`eval.py`](https://github.com/danruod/BERTPhylo/blob/main/eval.py) conta ins the model evaluation pipeline.
 * [`utils/`](https://github.com/danruod/BERTPhylo/tree/main/utils) directory contains the necessary files for the `eval.py` execution.


## Contact
Feel free to contact me (Danruo DENG: [drdeng@link.cuhk.edu.hk](mailto:drdeng@link.cuhk.edu.hk)) if anything is unclear or you are interested in potential collaboration.
