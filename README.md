# Lung-Infection-Classification-using-Convolutional-Neural-Networks
Artificial Intelligence has evolved a lot and is currently able to solve problems that are very complex and require human specialization. One such area is healthcare. A lot of research happens every day to use deep learning for the betterment of humanity, and one such is healthcare.
## Objective
To build a model using a convolutional neural network that can classify lung infection in a person using medical imagery

## Dataset Description
The dataset contains three different classes, including healthy, type 1 disease, and type 2 disease.

- Train folder: This folder has images for training the model, which is divided into subfolders having the same name as the class.
- Test folder: This folder has images for testing the model, which is divided into subfolders having the same name as the class.
## Tasks
Following operations should be performed using Keras or PyTorch or Torch vision:

- Import the necessary libraries
- Plot the sample images for all the classes
- Plot the distribution of images across the classes
- Build a data augmentation for train data to create new data with translation, rescale and flip, and rotation transformations. Rescale the image at 48x48
- Build a data augmentation for test data to create new data and rescale the image at 48x48
- Read images directly from the train folder and test folder using the appropriate function
- Build 3 CNN model with:

## CNN Architecture
- Add convolutional layers with different filters, max pool layers, dropout layers, and batch normalization layers
- Use Relu as an activation function
- Take the loss function as categorical cross-entropy
- Take rmsprop as an optimizer
- Use early stopping with the patience of two epochs and monitor the validation loss or accuracy
- Try with ten numbers epoch
- Train the model using a generator and test the accuracy of the test data at every epoch
- Plot the training and validation accuracy, and the loss
- Observe the precision, recall the F1-score for all classes for both grayscale and color models, and determine if the model’s classes are good
## Transfer learning using mobile net
- Prepare data for the pre-trained mobile net model, with color mode as RGB
- Create an instance of a mobile net pre-trained model
- Add dense layer, dropout layer, batch normalization layer on the pre-trained model
- Create a final output layer with a SoftMax activation function
- Change the batch size activation function and optimize as rmsprop and observe if the accuracy increases
- Take the loss function as categorical cross-entropy
- Use early stopping with the patience of two epoch and call back function for preventing overfitting
- Try with ten numbers epoch
- Train the model using a generator and test the accuracy of the test data at every epoch
- Plot the training and validation accuracy, and the loss
- Observe the precision, recall the F1-score for all classes for both grayscale and color models, and determine if the model’s classes are good
## Transfer Learning using Densenet121
- Prepare the dataset for the transfer learning algorithm using Densenet121 with the image size as 224x224x3
- Freeze the top layers of the pre-trained model
- Add a dense layer at the end of the pre-trained model followed by a dropout layer and try various combinations to get an accuracy
- Add the final output layer with a SoftMax activation function
- Take loss function as categorical cross-entropy
- Take Adam as an optimizer
- Use early stopping to prevent overfitting
- Try with 15 number of epoch and batch size with seven, also try various values to see the impact on results
- Train the model using the generator and test
