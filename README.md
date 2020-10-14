# mnist_dataset
# data transform
A small program for processing the mnist data set, using the example network, with an accuracy of 99%.
After processing the edge detection of the data set, the accuracy is 98%.
The main program is modified on the basis of https://zhuanlan.zhihu.com/p/95701775.
The network adopts https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py.



# tools
some .sh and .py files that are used to transform dataset.

add_0.sh:change pictures' names from "0.jpg" to "000.jpg"("34.jpg" to "034.jpg").

count_no_anno:count how many videos franmes without annos are over 50%.

