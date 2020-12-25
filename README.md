# F2GAN: Few-Shot Image Generation

Code for our **ACM MM 2020** paper *"F2GAN: Fusing-and-Filling GAN for Few-shot Image Generation"*.

Created by [Yan Hong](https://github.com/hy-zpg),  [Li Niu\*](https://github.com/ustcnewly), Jianfu Zhang, Liqing Zhang.

Paper Link: [[arXiv]](https://arxiv.org/pdf/2008.01999.pdf)



If you find our work useful in your research, please consider citing:

```
@inproceedings{HongF2GAN,
  title={F2GAN: Fusing-and-Filling GAN for Few-shot Image Generation},
  author={Hong, Yan and Niu, Li and Zhang, Jianfu and Zhao, Weijie and Fu, Chen and Zhang, Liqing},
  booktitle={ACM International Conference on Multimedia},
  year={2020}
}
```

## Introduction

Few-shot image generation aims at generating images for a new category with only a few images, which can benefit a wide range of downstream category-aware tasks like few-shot classification. In this paper, we propose a novel fusing-and-filling GAN (F2GAN) to enhance the ability of fusing conditional images. The high-level idea is fusing the high-level features of conditional images and filling in the details of generated image with relevant low-level features of conditional images. In detail, our method contains a fusion generator and a fusion discriminator. In our generator, we interpolate the high-level bottleneck features of multiple conditional images with random interpolation coefficients. Then, the fused high-level feature is upsampled through the decoder to produce a new image. In each upsampling stage, we borrow missing details from the skip-connected shallow encoder block by using our Non-local Attentional Fusion (NAF) module. In the fusion discriminator, we employ typical adversarial loss and classification loss to enforce the generated images to be close to real images and from the same category of conditional images. To ensure the diversity of generated images, we additionally employ a mode seeking loss and an interpolation regression loss, both of which are related to interpolation coefficients. We have conducted extensive generation and classification experiments on five datasets to demonstrated the effectiveness of our method.

![](figures/framework.png)


## Visualization 
More generated reuslts to view [here](https://arxiv.org/pdf/2008.01999.pdf)
![](figures/combo.jpg)





## Poster Presentation
![](figures/poster.png)



## Experiments

### Hardware& Software Dependency
- Hardware
    
  ** a single GPU or multiple GPUs
  
- Software

  ** Tensorflow-gpu (version >= 1.7)
  
  ** Opencv
  
  ** scipy  
  
- Click [here](https://github.com/bcmi/F2GAN-Few-Shot-Image-Generation/blob/main/requirements.txt) to view detailed software dependency            
      
  

### Datasets Preparation 
* The Download links can be found [here](https://github.com/bcmi/Awesome-Few-Shot-Image-Generation#Datasets)
- Omniglot
    
  **Categories/Samples:** 1412/ 32460
  
  **Split:** 1200 seen classes, 212 unseen classes
 
- Emnist

  **Categories/Samples:** 38/ 106400
  
  **Split:** 28 seen classes, 10 unseen classes
  
  
- VGGFace

  **Categories/Samples:** 2299/ 229900
  
  **Split:** 1802 seen classes, 497 unseen classes
  
- Flowers

  **Categories/Samples:** 102/ 8189
  
  **Split:** 85 seen classes, 17 unseen classes
      
- Animal Faces

  **Categories/Samples:** 149/ 214105
  
  **Split:** 119 seen classes, 30 unseen classes



### Baselines

#### Few-shot Image Generation

* FIGR: Few-shot Image Generation with Reptile [paper](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)  [code](https://github.com/sbos/gmn)

* Few-shot Generative Modelling with Generative Matching Networks [paper](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)  [code](https://github.com/sbos/gmn)

* DAWSON: A domain adaptive fewshot generation framework [paper](https://arxiv.org/pdf/2001.00576.pdf) [code](https://github.com/LC1905/musegan/)

* Data Augmentation Generative Adversarial Networks [paper](https://arxiv.org/pdf/1711.04340)  [code](https://github.com/AntreasAntoniou/DAGAN)

* Matchinggan: Matching-Based Few-Shot Image Generation [paper]((https://arxiv.org/pdf/2003.03497.pdf) [code](https://github.com/bcmi/MatchingGAN-Few-Shot-Image-Generation)


#### Few-shot Image Classification
* Matching Networks for One Shot Learning [paper](https://arxiv.org/pdf/1606.04080.pdf)  [code](https://github.com/AntreasAntoniou/MatchingNetworks)

* Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [paper](https://arxiv.org/pdf/1703.03400.pdf)  [code](https://github.com/cbfinn/maml)

* Learning to Compare: Relation Network for Few-Shot Learning [paper](https://arxiv.org/pdf/1711.06025.pdf)  [code](https://github.com/floodsung/LearningToCompare_FSL)


* Meta-Transfer Learning for Few-Shot Learning [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)  [code](https://github.com/y2l/meta-transfer-learning-tensorflow)

* Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation [paper](https://arxiv.org/pdf/2001.08735.pdf)  [code](https://github.com/y2l/meta-transfer-learning-tensorflow)

* Revisiting local descriptor based image-to-class measure for few-shot learning [paper](https://arxiv.org/pdf/1903.12290.pdf) [code](https://github.com/WenbinLee/DN4)



### Getting Started

### Installation

1.Clone this repository.

```
git clone https://github.com/bcmi/F2GAN-Few-Shot-Image-Generation.git
```

2.Create python environment for *F2GAN* via pip.

```
pip install -r requirements.txt
```




### Training

1.Train on Omniglot dataset

```
python train_dagan_with_matchingclassifier.py --dataset omniglot --image_width 96 --batch_size 20  --experiment_title MMF2GAN/omniglot1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 1 --loss_sim 1 
```

2.Train on EMNIST dataset

```
python train_dagan_with_matchingclassifier.py --dataset emnist --image_width 96 --batch_size 20  --experiment_title MMF2GAN/emnist1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 1 --loss_sim 1  
```

3.Train on VGGFce dataset

```
python train_dagan_with_matchingclassifier.py --dataset vggface --image_width 96 --batch_size 20  --experiment_title MMF2GAN/vggface1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 1 --loss_sim 1  
```



4.Train on Flowers dataset

```
python train_dagan_with_matchingclassifier.py --dataset flowers --image_width 96 --batch_size 20  --experiment_title MMF2GAN/flowers1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 1 --loss_sim 1  
```



5.Train on Animal Faces dataset

```
python train_dagan_with_matchingclassifier.py --dataset animals --image_width 96 --batch_size 20  --experiment_title MMF2GAN/animals1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 1 --loss_matching_D 1 --loss_sim 1  
```



### Trained Model

* [Omniglot](https://pan.baidu.com/s/1TtnqYaGaZmcMaDyHFLwrUw) extract code:5qab
* [EMNIST](https://pan.baidu.com/s/1cqnD5Ts8qjEN2DV0wPOGIg) extract code:sptr
* [VGGFace](https://pan.baidu.com/s/15w21JAw2uZ0xKRDVq5VuHg) extract code: 7hxu
* [Flowers](https://pan.baidu.com/s/1cFux1OQSrsX_UGUw0bCxPQ) extract code:km89
* [Animal Faces](https://pan.baidu.com/s/1puRJvBTjRtDQ7uu-isiyWQ) extract code:qe4h



### Evaluation from three aspects including GAN metrics, low-data classification, and few-shot classification.

#### 1. Visualizing the generated images based on trained models, the generated images are stored in the path '--experiment_title'

**Omniglot generated images**
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 3 --dataset omniglot --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_omniglot_MM --selected_classes 1 --support_number 3   --restore_path  ./trained_models/omniglot/ --continue_from_epoch 100
```

**EMNIST generated images**
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 3 --dataset emnist --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_emnist_MM --selected_classes 1 --support_number 3   --restore_path   ./trained_models/emnist/  --continue_from_epoch 100
```

**VGGFace generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 3 --dataset vggface --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_vggface_MM --selected_classes 1 --support_number 3   --restore_path    ./trained_models/vggface/  --continue_from_epoch 100
```


**FLowers generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 3 --dataset flowers --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_flowers_MM --selected_classes 1 --support_number 3   --restore_path    ./trained_models/flowers/  --continue_from_epoch 100
```


**Animal Faces generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 3 --dataset animals --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_animals_MM --selected_classes 1 --support_number 3   --restore_path  ./trained_models/animals/  --continue_from_epoch 100
```




#### 2. Testing the GAN metrics including FID and IPIPS for generated images, which is suitable for RGB 3-channel images like VGGFace, Flowers, and Animal Faces datasets.


**VGGFace GAN metrics**


```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/vggface/MM/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/vggface/MM/visual_outputs_forquality/  --image_width 96 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/vggface/MM/visual_outputs_forquality/ --out ./EVALUATION/Augmented/vggface/MM/GAN_METRICS.txt 

```


**Flowers GAN metrics**


```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/flowers/MM/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/flowers/MM/visual_outputs_forquality/  --image_width 96 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/flowers/MM/visual_outputs_forquality/ --out ./EVALUATION/Augmented/flowers/MM/GAN_METRICS.txt 

```


**Animals Faces GAN metrics**


```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/animals/MM/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/animals/MM/visual_outputs_forquality/  --image_width 96 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/animals/MM/visual_outputs_forquality/ --out ./EVALUATION/Augmented/animals/MM/GAN_METRICS.txt 

```




#### 3. Testing the classification in low-data setting with augmented images.
take Omniglot as example, low-data classification with augmented images generated from our trained model

3.1. Gnerating augmented images using three conditional images
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset omniglot --image_width 96  --batch_size 30  --num_generations 512 --experiment_title EVALUATION_Augmented_omniglot_MM --selected_classes 1 --support_number 3   --restore_path path ./trained_models/omniglot/ --continue_from_epoch 100
```

3.2. Preparing generated images: the generated images are stored in the 'storepath/visual_outputs_forclassifier' and setting the storepath for preprocessed data, running below script

```
python data_preparation.py --dataroot storepath/visual_outputs_forclassifier  --storepath --image_width 96 --channel 1 
```
3.3. Replacing the datapath in data_with_matchingclassifier_for_quality_and_classifier.py with the storepath for preprocessed data.

3.4. Running the script for low-data classification.

```
train_classifier_with_augmented_images.py --dataset omniglot  --selected_classes testing_categories  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedLowdataClassifier_omniglot  --image_width 96  --image_height 96 --image_channel 1
```
--selected_classes: the number of total testing categories


#### 4. Testing the classification in few-shot setting with augmented images.
take Omniglot as example, NwayKshot classification with augmented images generated from our trained model

4.1. Gnerating augmented images using Kshot conditional images
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset omniglot --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_omniglot_MM --selected_classes 1 --support_number K   --restore_path path ./trained_models/omniglot/ --continue_from_epoch 100
```
setting the '--support_number' as K.

4.2. Preprocessing the generated images
```
python data_preparation.py --dataroot ./EVALUATION/Augmented/omniglot/MM/visual_outputs_forclassifier  --storepath ./EVALUATION/Augmented/omniglot/MM/  --image_width 96 --channel 1 
```

4.3. Replacing the datapath in data_with_matchingclassifier_for_quality_and_classifier.py with ./EVALUATION/Augmented/omniglot/MM/omniglot.npy.

4.4. Running the script for few-shot classification.

```
train_classifier_with_augmented_images.py --dataset omniglot  --selected_classes N  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedFewshotClassifier_omniglot --image_width 96  --image_height 96 --image_channel 1
```
setting the '--selected_classes' as N.





### Results

#### Learned Attention
![](figures/query_attention.png)



#### GAN metrics of Generated Images

![](figures/metric.jpg)


#### Low-data Image Classification

![](figures/lowdata.jpg)



#### Few-shot Image Classification

![](figures/fewshot.jpg)





## Acknowledgement

Some of the codes are built upon [DAGAN](https://github.com/AntreasAntoniou/DAGAN). Thanks them for their great work!

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!

*F2GAN* is freely available for non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.



