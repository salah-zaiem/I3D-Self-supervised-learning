# I3D-Self-supervised-learning
Training the I3D Architecture (Carreira and Zissermann) on a proxy task in a self supervised setting

In our project we implement the Odd-One-Out (O3N) proxy task and a novel proxy task. 
These self-supervised tasks are used in order to pre-train an I3D network on the UCF-101 dataset.
To test the benefits of this pre-training we extract features from the I3D network and use them to classify videos from the UCF-101 dataset.
We compare the results of this classification with the features extracted from the original Kinetics pre-trained I3D network and from the I3D network pre-trained using the two proxys.  
