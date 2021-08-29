# Artifiacl-Neural-Network-Algorithm
Supervised Machine Learning Algorithm - Artificial Neural Network Algorithm

The neural artificial network is essentially a multi-layer architecture. Every layer has some information that is used to forward the information to the next layer. Therefore, layers and architecture for building ANN must be defined. 

Here, a four layer sequential model of forward propagation is being used as architecture. One also needs to choose an activation function for defining each layer. Activation functions are extremely essential for introducing non-linearity into the model in neural networks. A number of options are available to choose the activation function from [1]. For the first three hidden layers, I used ReLU as an activation function. The fact that ReLU limits processing of models in case it is not necessary is the reason why this activation function is selected. Also, it is easier to calculate i.e.f(x)=max(0,x) {f(x)= output function}[1] and takes less time to process data which is another factor for selection. 

Softmax is used as an activation function in the output layer because it works best for multi-class classification.  Softmax gives probability of a datapoint and assigns decimal probability to each class of selected application. The next step involved in selecting the loss function. I used catagorical_crossentropy to optimise the occurrence of any kind of tolerance in the results. Then an optimiser “adam” is introduced to reduce losses and adjust the attributes in such a way that weights and learning rate contribute to least losses. Adam optimiser is good in handling big and complex dataset and uses less memory.

Confusion matrix is created to diagnose the performance of classifiers. The evaluation matrices such as accuracy, precision, recall and F1 score are also used to evaluate the performance of classifier models which will be discussed in the next section.
