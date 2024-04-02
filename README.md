# :book: UWFormer
[**UWFormer: Underwater Image Enhancement via a Semi-Supervised Multi-Scale Transformer**
](https://arxiv.org/abs/2310.20210)

**Weiwen Chen, Yingtie Lei, [Shenghong Luo](https://shenghongluo.github.io), Ziyang Zhou, Mingxian Li and [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun)**

**University of Macau**

# :mag: Usage

## Installation
```
git clone https://github.com/leiyingtie/UWFormer.git
cd UWFormer
pip install -r requirements.txt
```
## Training
**Specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in**`config.yml`.

**For single GPU training:**
```
python train.py
```

**For multiple GPUs training:**
```
accelerate config
accelerate launch train.py
```
**If you have difficulties with the usage of** `accelerate`, **please refer to** [Accelerate](https://github.com/huggingface/accelerate).

# 💗 Acknowledgements
This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grant 0087/2020/A2 and Grant 0141/2023/RIA2.

# :bulb: Citation
**If you find our work helpful for your research, please cite:**
```bib

```
