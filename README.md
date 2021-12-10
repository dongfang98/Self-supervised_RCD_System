# Self-supervised_RCD_System
Recommendation System  
https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/embed_anim.mp4
## 1.	Generate the label for the samples
In this project, we use greedy modularity maximization, one of the modules can use the graph as input, and divided the nodes in graph to different clusters. We use different colors to denote different clusters, that is, if two nodes are both red, that means they are in the same cluster. This method is used to generate the true label of our training sample. 

Here’s part of greedy modularity maximization’s principle:
Find communities in G using greedy modularity maximization. This function uses Clauset-Newman-Moore greedy modularity maximization. This method currently supports the Graph class. Greedy modularity maximization begins with each node in its own community and joins the pair of communities that most increases modularity until no such pair exists. (Reference:https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)

## 2. Layers in the neural network
As for the input layer and hidden layers, we used the GCN layers’ update function to update each layer:
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/GCN_Layer.png)

Because every neural network layer can be written as a non-linear function. The A denotes the adjacent matrix of the graph. So, we can get the l+1 layer’s neural network by l layer and the adjacency matrix. 
(Reference: https://tkipf.github.io/graph-convolutional-networks/)

![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/NeuralNetwork.png)

As for the output layer, we used the Softmax layer. Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/Softmax_Layer.png)

## 3. Gradient checking in every layers

We need to do the gradient checking during the back propogation process to ensure the correctness: 
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/GradientChecking.png)

The equation above is basically the Euclidean distance normalized by the sum of the norm of the vectors. We use normalization in case that one of the vectors is very small.
As a value for epsilon, we usually opt for 1e-7. Therefore, if gradient check return a value less than 1e-7, then it means that backpropagation was implemented correctly. Otherwise, there is potentially a mistake in your implementation. If the value exceeds 1e-3, then you are sure that the code is not correct.
(reference:https://towardsdatascience.com/how-to-debug-a-neural-network-with-gradient-checking-41deec0357a9)

## 4. Start training

We should input the quantity of the nodes, the quantity of classes we expect, the quantity of hidden layer, the size of the hidden layers and the activation function. In this project, we use the sigmoid function as our activation function. We used the gradient descent in back propogation to adjust the W. Here's our training loss picture:
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/Training.PNG)

## 5. Final result format
After GCN, the picture is like this. As you can see, the nodes with same color will form a cluster, the closer they are, the more similar they are. 
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/AfterGCN.PNG)

This is our reference code: https://github.com/zjost/blog_code/tree/master/gcn_numpy
