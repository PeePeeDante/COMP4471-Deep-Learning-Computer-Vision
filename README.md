## COMP4471-Deep-Learning-Computer-Vision

## pa1: KNN, SVM, Softmax, Neural Network
  Raw Numpy implementation of ML algorithms + Numpy Vectorization for performance optimization </br>
  Dataset: run cs231n/datasets/get_datasets.py

## pa2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets
  Raw Numpy implementation of Batch Normalization, Dropout, CNN + Numpy Vectorization for performance optimization </br>
  Dataset: run cs231n/datasets/get_datasets.py
  
## pa3: Vanilla RNN, LSTM, RNN on Language Models, Image Captioning, GAN, Style Transfer, Network Visualisation
  Raw Numpy implementation of RNN and apply them for various applications </br>
  Dataset: run cs231n/datasets/get_assignment3_data.py

## Final Project: Day-night Image Transformation Using CycleGAN and D2-net
### Abstract 
  Illumination variability is arguably one of the most crucial factors in image matching, a task in which aligning structure, pattern, and content between photos remain a hurdle to overcome. A sharp difference in illumination between images can significantly compromise the matching performance, in particular the matching of the images taken at day and those at night. In this study, we propose a GAN-based, image-to-image translation network to tackle the day-night feature matching challenge. Our modified CycleGAN model transforms day images to night ones and vice versa, followed by analysis using D2-net descriptors. By comparing our modified CycleGAN model to the vanilla CycleGAN and the OpenCV models, the proposed model produces greater image quality and delivers better performance on feature matching.

### Proposed Algorithm

An intuitive algorithm that performs day-to-night (or night-to-day) transformation is proposed in this study, in which the transformed images and the other night/day photos will serve as the input for the downstream matching task. For the image transformation stage, a CycleGAN model is used for training (with some detailed modifications, please refer to the report pdf) since it is known for learning style transferring between images. Its strength is exemplified by mapping horses to zebras, Van Gogh’s painting to Monet’s, and, in our case, day images to night images. Our CycleGAN model will be trained on the day images and night images from the 24/7 Tokyo dataset \[10\]. For the feature matching algorithm, the pre-trained D2-net model is chosen, given its excellent performance in day-night localization tasks. The Aachen Day-Night dataset \[11\] is used for our final evaluation, where a night query image is paired with a transformed-day image and a transformednight query is paired with a day image. In the end, the results from the two pipelines will be aligned, and those with a matching score above a given threshold will be retained. </br>

<p align="center">
  
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/cyclegan_transform.jpeg"  width="500">
</br>
CycleGAN 
</br>
</br>

<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/D2net.png"  width="500"> 
</br>
D2Net
</br>
</br>

<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/pipeline.png"  width="300"> 
</br>
Algorithm Pipeline
</br>

</p>
</br>

### GAN Training Results (Day -> Night)

\[Training 400 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-000400-Y-X.png"  width="300"> 

\[Training 2000 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-002000-Y-X.png"  width="300"> 

\[Training 3600 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-003600-Y-X.png"  width="300"> 

\[Training 6400 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-006400-Y-X.png"  width="300"> 

\[Training 13600 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-013600-Y-X.png"  width="300"> 
</br>

### GAN Results vs OpenCV brightness 

This section shows the generated image comparison from GAN compared with brightness modification using OpenCV.

</br>
<p align="center">
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/result_compare.png"  width="700"> 
</br>
Generated image comparison
</br>
  
</p>

### Feature Matching Results

### Reference
  
