# Visual Contrastive Learning for Few-shot Image Classification
This repository provides the code to classify images in two different categories, i.e. Similar (1) and Dissimilar (0) based on the image similarity task performed by utilizing a Contrastive Learning-based approach (including employing a custom contrastive loss). Furthermore, [Siamese Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) is being used in n-way k-shot settings considered in the current implementation.
## Requirements
- `Python 3.9`
- `PyTorch 1.10.2`
- `TorchVision 0.11.3`
- `numpy 1.22.3`
- `matplotlib 3.5.1`
## Usage
### Data
[Omniglot](https://github.com/brendenlake/omniglot/tree/master/python) dataset is being used which is a collection of 1623 hand drawn characters from 50 different alphabets. For every character there are just 20 examples, each drawn by a different person. Each image is a gray scale image of resolution 105x105. Please clone [this](https://github.com/brendenlake/omniglot/tree/master/python) repo and then extract the `images_background` and `images_evaluation` folders. Finally, run `DataGeneration.py` file to create pickle files `train.pickle` and `val.pickle` files and store them in `data` folder. Here, `train.pickle` file contains characters from 30 different alphabets, whereas `val.pickle` contains characters from remaining 20 different alphabets.
### Model Building and Training
- The `SiameseNetwork` model class for n-way k-shot learning can be found in `Model.py` file.
- To train the network, run `Training.py` file.
- The average loss for the trained model is printed after every epoch.
- All hyperparamters to control training and testing of the model are provided in the given `Training.py` file.
## Output Samples
### Image Similarity Scores
| Image Comparison 1        | Image Comparison 2           | Image Comparion 3           |
| ------------------------- |:----------------------------:|:---------------------------:|
| ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/1.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/2.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/3.png) |

| Image Comparison 4        | Image Comparison 5           | Image Comparion 6           |
| ------------------------- |:----------------------------:|:---------------------------:|
| ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/4.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/5.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/6.png) |

| Image Comparison 7        | Image Comparison 8           | Image Comparion 9           | Image Comparison 10           |
| ------------------------- |:----------------------------:|:------------------:|:-------------------:|
| ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/7.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/8.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/9.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/10.png) |
### Results for Image Classification
| Image Comparison 1        | Image Comparison 2           | Image Comparion 3  |
| ------------------------- |:----------------------------:|:------------------:|
| ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/11.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/12.png) | ![alt text](https://github.com/fork123aniket/Visual-Contrastive-Learning-for-Few-shot-Image-Classification/blob/main/Images/13.png) |
### Analysis
Among all the 10 comparisons made under ***Image Similarity Scores*** sub-section, images 1, 6, and 8 appear more similar, thereby having predicted labels as 1, as shown in the ***Results for Image Classification*** sub-section. This way, the current implementation frames the image similarity task as the image classification task.
