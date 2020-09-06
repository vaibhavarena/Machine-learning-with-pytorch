# Machine-learning-with-pytorch

<h3>60min.py</h3>
<p>Use of CIFAR 10 data to train a model using CrossEntropyLoss as loss function and SGD as optimizer. Used a basic custom neural network.</p>
<br>

<h3>fashion.py</h3>
<p>Training the Fashion MNIST data on a custom model. Training loss, Testing loss and accuracy are given as outputs.
A graph for showing variation between training and testing loss over time.</p>
<p>Use of NLLLoss along with log_softmax for the loss function.
SGD optimizer has been used in which the complete training set is divided in some batches and weights are updated accordingly,
which helps when the dataset is larger and SGD demands less resources. Adam optimizer is like SGD with momentum and automatic tweaks
to learning rate for different weights.</p>
