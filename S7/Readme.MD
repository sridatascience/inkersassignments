          
          Requirements:
                    Change the code such that it uses GPU
                    Change the architecture to C1C2C3C40 (3 MPs)
                    Total RF must be more than 44
                    One of the layers must use Depthwise Separable Convolution
                    One of the layers must use Dilated Convolution
                    Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
                    Achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M

          Results:
                    RF > 44
                    Epochs: 25
                    Accuracy > 80% after 14th Epoch
                    Added GAP
                    Added Depthwise Separable Convolution
                    Added Dilated Convolution
