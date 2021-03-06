# GCN_Recommendation_System
This is a movie recommendation system based on GCN (graph convolutional network).  
To have a test, run movie_rcd.py, all the datasets are in fold ml-latest-small.  
The flow chart below shows what we have done:  
![image](https://user-images.githubusercontent.com/78338843/145574755-51fed9af-df63-4c6c-b7a9-405602f1ac83.png)

## 1. Dataset
1. Filter: rating == 5  
2. Merge: find users share the same 5-rating movies  
Then generate graph by adding edges between users according to User-User Table  
![image](https://user-images.githubusercontent.com/78338843/145575111-4379aead-b203-424a-b4ea-243923ba01f4.png)

## 2. Method(GCN)
Procedure of training and testing:
![image](https://user-images.githubusercontent.com/78338843/145574087-c6e46904-49a8-4cb8-8a8e-8e1fa21a65d2.png)

### 2.1	Generate the label for the samples
Firstly, we use greedy modularity maximization, one of the modules can use the graph as input, and divided the nodes in graph to different clusters. We use different colors to denote different clusters, that is, if two nodes are both red, that means they are in the same cluster. This method is used to generate the true label of our training sample. 

Here’s part of greedy modularity maximization’s principle:
Find communities in G using greedy modularity maximization. This function uses Clauset-Newman-Moore greedy modularity maximization. This method currently supports the Graph class. Greedy modularity maximization begins with each node in its own community and joins the pair of communities that most increases modularity until no such pair exists. (Reference:https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)

### 2.2 Layers in the GCN
As for the input layer and hidden layers, we used the GCN layers’ update function to update each layer:
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/GCN_Layer.png)

Because every neural network layer can be written as a non-linear function. The A denotes the adjacent matrix of the graph. So, we can get the l+1 layer’s neural network by l layer and the adjacency matrix. 
(Reference: https://tkipf.github.io/graph-convolutional-networks/)

![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/NeuralNetwork.png)

As for the output layer, we used the Softmax layer. Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.

![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/Softmax_Layer.png)

### 2.3 Gradient checking in every layers

We need to do the gradient checking during the back propogation process to ensure the correctness: 
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/GradientChecking.png)

The equation above is basically the Euclidean distance normalized by the sum of the norm of the vectors. We use normalization in case that one of the vectors is very small.
As a value for epsilon, we usually opt for 1e-7. Therefore, if gradient check return a value less than 1e-7, then it means that backpropagation was implemented correctly. Otherwise, there is potentially a mistake in your implementation. If the value exceeds 1e-3, then you are sure that the code is not correct.

(reference:https://towardsdatascience.com/how-to-debug-a-neural-network-with-gradient-checking-41deec0357a9)

### 2.4 Start training

We should input the quantity of the nodes, the quantity of classes we expect, the quantity of hidden layer, the size of the hidden layers and the activation function. In this project, we use the sigmoid function as our activation function. We used the gradient descent in back propogation to adjust the W. Here's our training loss picture:

![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/Training.PNG)
 
Here is a video of the training process:  
https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/embed_anim.mp4

## 3. Result & Conclusion
As is shown in the picture below, after GCN, nodes with same color are separated in the same cluster, the closer they are, the more similar they are. We use the position of nodes in the graph to calculate the distance between the target user and other users. Then we select the user with the smallest distance as similar user and recommend his/her favorite movies.  
![image](https://user-images.githubusercontent.com/78338843/145572885-2f733253-0047-4628-9e3d-a2294be0409d.png)

## 4. Reference
This is our reference code: https://github.com/zjost/blog_code/tree/master/gcn_numpy
