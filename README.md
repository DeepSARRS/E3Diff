<!--
 * @Description: 
 * @Date: 2024-11-23 12:26:20
 * @LastEditTime: 2024-11-23 16:41:34
 * @FilePath: /QJ/E3Diff/README.md
-->
# Efficient End-to-end Diffusion Model for Onestep SAR-to-Optical Translation

### 🚀 E3Diff won the 1st in CVPR PBVS2025 Multi-modal Aerial View Image Challenge - T (Translation) 🎉

## Brief
This is an official implementation of **[Efficient End-to-end Diffusion Model for Onestep SAR-to-Optical Translation (E3Diff)](https://ieeexplore.ieee.org/document/10767752)** by **PyTorch**.
- [√] released dataset and weights
- [√] log / logger
- [√] metrics evaluation
- [√] multi-gpu support
- [√] resume training / pretrained model
- [√] [Weights and Biases Logging]
- [√] 1/multi steps training and sampling
- [√] SEN12 results of baseline methods are released
  
## Pipeline
![vis](/doc/pipeline.png)

## Result of SEN12 Dataset
![vis](/doc/sen12_vis.png)

## Result of SAR2EO Dataset
![vis](/doc/sar2eo_vis.png)




## Usage
### Environment
- create a new environment:
```bash
$ conda env create -f environment.yml

```

- install [softpool](https://github.com/alexandrosstergiou/SoftPool).
```bash
$ cd SoftPool/pytorch
$ make install
--- (optional) ---
$ make test
```


### Dataset
SAR-EO dataset: [baiduyun code 0615](https://pan.baidu.com/s/19aPkwppi5WeBy8j18aJ4-w?pwd=0615)

SEN12 dataset: [google drive](https://drive.google.com/drive/folders/1KZMXgHsXUuztxPI44jKeFj29KYHLbopP?usp=sharing)

### Training:
Download the dataset, and train your model using the following commands (about 1 week using 2 A6000 48GB GPU):


```bash
# stage 1 training for sen12 dataset (PPB filtering is not used for SEN12)
python main.py --config 'config/SEN12_256_s1.json'

# stage 2 training for sen12 dataset (PPB filtering is not used for SEN12)
python main.py --config 'config/SEN12_256_s2_1step.json'

```

Also, you might be willing to download the well-trained model of SEN12 from [here](https://drive.google.com/drive/folders/1KZMXgHsXUuztxPI44jKeFj29KYHLbopP?usp=sharing), and test the model:
If needed, results of some baseline methods can also be downloaded [here](https://pan.baidu.com/s/19OgulOKHcbVujgoBVIx6vw?pwd=tf95)(code is tf95).


```bash
# stage 2 validation for sen12 dataset
python main.py --config 'config/SEN12_256_s2_test.json' --phase 'val'  --seed 1
```

```bash
If you want to reproduce results of SAR2EO, please use the matlab code 'SAR2EO_filter.m' to filter speckles of SAR images before feeding into E3Diff.
```
## 🚀 Weights and Biases 🎉

The library now supports experiment tracking, model checkpointing and model prediction visualization with [Weights and Biases](https://wandb.ai/site). You will need to [install W&B](https://pypi.org/project/wandb/) and login by using your [access token](https://wandb.ai/authorize). 

```
pip install wandb

# get your access token from wandb.ai/authorize
wandb login
```




## Acknowledgements

Our work is mainly based on the following projects:

- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
- https://github.com/GaParmar/img2img-turbo
- https://github.com/alexandrosstergiou/SoftPool



## Citation
If you find the project useful, please cite the papers:



```
@ARTICLE{10767752,
  author={Qin, Jiang and Zou, Bin and Li, Haolin and Zhang, Lamei},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Efficient End-to-End Diffusion Model for One-step SAR-to-Optical Translation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LGRS.2024.3506566}}

@article{qin2024conditional,
  title={Conditional Diffusion Model with Spatial-Frequency Refinement for SAR-to-Optical Image Translation},
  author={Qin, Jiang and Wang, Kai and Zou, Bin and Zhang, Lamei and van de Weijer, Joost},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}

```
