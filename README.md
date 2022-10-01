# Time series prediction using NARX net
Time series prediction example using NARX network. 
a synthetic data is used to train the NARX net.

In this example, open-loop architecture is used during the training step because of two
advantages. First, the resulting network is purely feedforward, and static backpropagation
is used for training. Secondly, open-loop uses the actual values as
input of the network and the result is more precise . open-loop can only be able to predict
one step ahead and to overcome this limitation, after the training of the NARX network
with open-loop architecture the network will be converted to the closed loop architecture to perform
multi-step-ahead prediction ahead.
