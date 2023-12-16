# GazeTracking
## Overview
  GazeTracking is a deep neural network project designed to predict the point of gaze on a computer screen by analyzing facial images. Notably, the achieved results are obtained using a relatively modest RGB camera (0.9 MP laptop webcam) without additional variations in illumination, such as an infrared light projector. 
  This technology holds numerous practical applications, with one notable example being its potential to assist individuals with various Motor Neuron Diseases in using electronic devices.

## Data Collection and Data Augmentation
  The dataset for this project was meticulously gathered in multiple sessions, each accompanied by a small solid circle traversing the screen horizontally from top to bottom. Images were captured at random distance intervals to ensure a diverse dataset, mitigating biases that might arise at the edges where the circle slows down.
  After obtaining the facial landmark coordinates using Mediapipe, the images were cropped to focus exclusively on the face. Subsequently, random cropping, horizontal image mirroring, and color transformation for YCbCr colorspace was applied to enhance the network's reliability, resulting in 224px square images. Additionally, the dataset includes a 25x25 binary face grid, indicating the position of all face pixels in the original captured image.

  *Final Dataset* contains of 4 inputs:
1. Face Image
2. Left Eye Image
3. Right Eye Image
4. Binary Grid

  It's worth noting that, due to the utilization of a **cloud GPU**, the extraction of eye images was performed in the cloud, significantly expediting the dataset uploading process—making it three times faster.


## Neural Network
  For RGB-based gaze-tracking, the iTracker [1] network architecture, recognized as the current state-of-the-art, was employed in my application with a slight modification. Unlike iTracker, which is based on AlexNet as the backbone architecture, the pre-trained ResNet was implemented instead. This modification was made to enhance the network's performance by adding skip connections.
  To address regularization, a combination of Image Scaling, Batch Normalization, and Dropout layers (with a 10% probability) were incorporated into the architecture. These techniques contribute to the stability and generalization of the model. To ensure proper tuning, *Cyclic Learning Rate* was applied. This approach practically eliminates the need for extensive hyperparameter tuning, providing an efficient and effective training methodology.
  
## Results
