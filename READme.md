This Data Pipeline Setup for DSGT competition

The pipeline is created to streamline testing various models.

Optimal number of batch results:

Logging Progress:
Find Optimal Learning Rate for adam
Plotting lr was epoch to find the approximate learning rate for all the models.

Initially, ResNet18 with pretrained weights achieved test loss in the submission of 0.86.

Moving to larger models such as ResNet34 and resNet50 resulting in worst train loss and test loss, it also shows clear signs of overfitting. 
Solution:
1) tried checking the distribution of the weights with histogram. Nothing was wrong, as the ResNet layers have batchNorm therefore the distribution of training data should be quite optimal.

Training ResNet18 Journal
The Data Augmentation was resulting in train loss converging but the test loss does not and remains the same.
Solution: 
1) Test to see it Data Augmentation is not representative of the Dataset, by testing if the model converges with the test set also being augmented. Founded that indeed the original Data Augmentation has its value too saturated causing the training to be misrepresentative of the test set. The model converged when the test set is also being augmented.
Next Step:
1) Add Horizontal Flip and view the results if this feature is representative of the data. Did not work.
2) Maybe the normalization is not working properly, try removing the normalization
Solution: I forgot to normalize my test_set.
Optimizing the Data Augmentation:
3) However, the test set is still not converging as well as expected.
4) After trying the SGD, the model converges in a very fast manner. Therefore, there is likely something wrong with my configuration of Adam.

Vision Transformers:
Takes very long to train.