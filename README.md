# A-Curious-FFN

Overview
--------
The project focuses on developing an FFN (Feed-Forward Network) model with a unique feature of "Artificial Curiosity". Artificial
Curiosity in the context of this project is a process that involves applying affine transformations to input test data and using a
model trained on regular; non-transformed data to predict on the transformed data. The model then filters the predictions based on
the least entropy for log-softmax values, specifically softmax values. The choice of the word "curiosity" to describe the process of
applying different transformations to test images makes sense, as it reflects the idea of exploring different possibilities to make
a prediction. The model is "curiosity-driven" in the sense that it explores different possibilities of orientations and mirroring of 
the test image in order to make a prediction, rather than relying solely on what it was trained on. This demonstrates some level of 
creativity in the sense that the model is able to find new ways of solving the problem of image recognition, rather than just being 
limited to what it was trained on. While the number and types of transformations used in this project are limited for demonstration 
purposes, the concept of artificial curiosity can be extended to include a range of creative transformations. This project uses a 
feed-forward neural network (FFN) because FFNs lack spatial perception and are unable to recognize images that do not resemble the 
training data. This weakness of FFNs is exploited to demonstrate the concept of artificial curiosity. When the model is predicting 
with 'curious' mode, it can recognize manipulated images, while in 'standard' mode, it cannot. This distinction provides a clear 
demonstration of the advantages and limitations of the two modes.

Table of Contents
-----------------
- Primary Objective
- Methodology
- Results
- Usage
- Setup
- Contributing
- Credits
- License
- Contact

Primary Objective
-----------------
The primary objective of this project is to develop a Feed-Forward Network (FFN) model with a unique feature of "Artificial Curiosity".
The project aims to show that the model is able to find new ways of solving the problem of image recognition, demonstrating some level
of creativity. The project also aims to demonstrate the advantages and limitations of the "curious" and "standard" modes of the model 
in recognizing manipulated images.

Methodology
-----------
1. **Establish connection between Git repository on Google Drive and GitHub using SSH protocol.**<br>
  This step facilitates working with Google Colab and provides access to resources like GPUs if required.
  
2. **Download MNIST dataset through scikit-learn library**<br>
  Load, inspect and prepare the data for Deep Learning.  
  
3. **Split the data into train and test sets.**<br>
  First convert NumPy arrays to pytorch Tensors, use Stratified Shuffle Split technique to split the data, then convert to pytorch
  datasets using TensorDataset, finally translate to DataLoader Objects.
  
4. **Create and train a Feed Forward Network.**<br>
  Define an Abstract Base Class called BaseCuriosity, create concrete class called CuriousFFN which inherits from the ABC and
  nn.Module, initialize attributes as defined by ABC. Modify the `__call__` method to allow 'standard' and 'curious' modes, define 
  forward (standard mode) and curiosity methods (curious mode), instantiate the class and train the model on non-manipulated data, 
  visualize the performance.
  
5. **Perform a few preliminary tests on standard and curious modes.**<br>
  Apply transformations to the test image and predict with both modes to validate whether the model is functional or not.
  
6. **Define versatility score as a performance measure for quantifying the impact of artificial curiosity on model performance.**<br>
  Versatility score is the difference between performance of the model in curious and non-curious modes. Minimum and Maximum value of versatility score are 0 and 1. Any of the standard performance measures like accuracy score or others can be used for calculating versatility score.
  
7. **Prepare the test images.**<br>
  Define rules for transformations to be applied on test data that complies with the scope of the defined curiosity, apply  transformation to the images in the testloader, as a preliminary check for conducting experiments predict using standard and curious
  modes, finally record the performance and calculate versatility.
  
8. **Conduct an experiment to test the impact of artificial curiosity on model performance.**<br>
  Pick 'n' random samples from manipulated images (Random Sampling with replacement), predict in standard mode and measure 
  performance, predict in curious mode and measure performance. Calculate versatility score for this iteration of the experiment, 
  repeat the iteration 50 times.
  
9. **Conduct a hypothesis test.**<br>
  Check the assumptions of t-test, test the null hypothesis that "The performance of the standard and curious modes in predicting
  the manipulated images is not significantly different" and then present the results of the experiment.
  
Results
-------
Accuracy score was used to calculate Versatility score, therefore in the experiment Versatility score is defined as the difference
between accuracy of curious and non-curious modes of the model. In standard mode the model achieved average accuracy of 0.25 while the curious mode achieved average accuracy of 0.61. This resulted in a Versatility score of 0.36, this implies with the help of artificial curiosity the model had an improvement of 36% compared to its standard classification ability, which is quite interesting. The null hypothesis was rejected due to negative t-statistic value of -133.767 and p-value being 9.50e-113 which is well the common significance level of 0.05.

Usage
-----
Thank you for your interest in this project, currently this repository is dedicated for demonstration of the concept only and the models trained in the notebook were not saved and as a result not available for use. However in the future we're likely to make the trained models with enhanced artificial curiosity available for use. If you would like to try building a model with artificial curiosity as explained above, please refer the notebook for details.
