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
  After the first successful training session, the neural network demonstrated remarkable progress, achieving a validation accuracy below 200 pixels, equivalent to approximately 5cm.   
       
<div align="center">
  <span>  <i> First Learning results </i>  </span>
  <p>
  <img src="https://github.com/majewski00/gaze-tracking/assets/153656493/f1ad9978-fd99-4407-8712-2ca8a93e4fb6" alt="image_1" width="800"/>
    </p>
  <br>
</div>   
  
  
Additionally, after performing *data analysis* on the worst predictions, it turns out that the majority of errors occur around the left and right edges of the screen. This observation indicates that the dataset, collected in a horizontal manner, lacks quality images in those areas.  
  
<div align="center">
  <span>  <i> Historgams and Heatmap of the first learning </i>  </span>
  <p>
  <img src="https://github.com/majewski00/gaze-tracking/assets/153656493/703485bf-ffe8-4bd5-be15-2f0830f1df91" alt="image_1" width="800"/>
    </p>
</div>  
<br>

## Current development  
- Applying low-precision (16 bit float) into Dataset  
- Testing the accuracy of PirateNets [3] instead of ResNets  
- Graphical interface, to display and test the result in real-time  
<br>
<p align="center">❕ ❕  WIP  ❕ ❕</p>
 
## References
[1] K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba  - **Eye tracking for everyone.**   
[2] J. Sharma, J. Campbell, P. Ansell, J. Beavers, and C. O’Dowd -   **Towards Hardware-Agnostic Gaze-Trackers.**  
[3] S. Wang, B. Li, Y. Chen, P. Perdikaris, 2024 - **PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks**  
