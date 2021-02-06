Assignment:

Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
Please make sure that your test_transforms are simple and only using ToTensor and Normalize
Implement GradCam function as a module. 
Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
Target Accuracy is 87%
Submit answers to S9-Assignment-Solution.

Submission:
        Parameters:
        Loss Function: Cross Entropy Loss
        L1 decay: 1e-6
        L2 decay: 1e-3
        Optimizer: SGD
        Learning Rate: 0.01
        Model: Resnet18
        Model Parameters: default parameters with dropout(0.1) added
        Epochs: 25
        
       
       
        Results
          Heighest Accuracy achieved: 87.85% (epoch:19)
          Observations:
          Consistently above 80% accuracy from epoch 7 onwards.
         
