# Ribonanza-RNA-Folding

Last year, Stanford hosted a competition aimed at predicting RNA base reactivity using Map and 2A3 reactivity data. 
To address this task, I used a conformer model architecture, an architecture which was invented for speech recognition, but it seems to work well with sequence data.

incorporating two embedding layers: one for encoding the RNA bases and another for positional information.

The model concludes with a multi-layer perceptron (MLP) layer to generate predictions. And since the reactivity values range from 0 to greater than 1, I employed the ReLU activation function to ensure all outputs remain non-negative.

For the loss function, I selected L1 loss, which evaluates the absolute difference between predicted and actual values.

Given the presence of numerous NaN values in the dataset, I replaced these with zeros and applied a masking technique to ensure they did not influence the training process.
