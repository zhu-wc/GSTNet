# Geometry-sensitive semantic modeling in visual and visual-language domains for image captioning

This repository contains the reference code for the paper [Geometry-sensitive semantic modeling in visual and visual-language
domains for image captioning](https://www.sciencedirect.com/science/article/pii/S0952197625003306?via%3Dihub)

![](https://github.com/zhu-wc/GSTNet/blob/main/images/overview.jpg)

## Experiment setup

Most of the previous works follow [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer), but they utilized some lower-version packages. Therefore, we recommend  referring to [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx). 

## Data preparation

* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing). Extarct and put it in the project root directory. Then, some preprocess follow [here](https://github.com/luo3300612/Transformer-Captioning/blob/main/pre_tokenize.py)
* **Feature**. We extract feature with the code in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). You can download the features we used [here](https://github.com/luo3300612/image-captioning-DLCT).
* **evaluation**. We use standard evaluation tools to measure the performance of the model, and you can also obtain it [here](https://github.com/luo3300612/image-captioning-DLCT). Extarct and put it in the project root directory.

## Training

```python
python train.py --exp_name test --batch_size 50 --head 8 --features_path coco_all_align.hdf5 --annotation_folder annotation --workers 8 --rl_batch_size 100 --image_field FasterImageDetectionsField --model transformer --seed 118 --rl_at 17
```

## Evaluation

```python
python eval.py --annotation_folder annotation --workers 5 --features_path coco_all_align.hdf5 --model_path saved_models/pretrained_model.pth
```
Pretrained model is available [here](https://drive.google.com/file/d/1CFKX2W-W_MgQjCE3ZPyp12Xc0SXk-zCS/view?usp=sharing)
## References

[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

[3] [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx)

## Acknowledgements

Thanks the original [m2](https://github.com/aimagelab/meshed-memory-transformer) provided the basic framework of the code. Thanks the author of [DLCT](https://github.com/luo3300612/image-captioning-DLCT) for optimizing the original code in [here](https://github.com/luo3300612/Transformer-Captioning) to  accelerate the training process, which has been of great assistance to  our research.
