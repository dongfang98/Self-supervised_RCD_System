# Self-supervised_RCD_System
Recommendation System  
https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/embed_anim.mp4
## 1.	Generate the label for the samples
In this project, we use greedy modularity maximization, one of the modules can use the graph as input, and divided the nodes in graph to different clusters. We use different colors to denote different clusters, that is, if two nodes are both red, that means they are in the same cluster. This method is used to generate the true label of our training sample. 

Here’s part of greedy modularity maximization’s principle:
Find communities in G using greedy modularity maximization. This function uses Clauset-Newman-Moore greedy modularity maximization. This method currently supports the Graph class. Greedy modularity maximization begins with each node in its own community and joins the pair of communities that most increases modularity until no such pair exists. (Reference:https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)

## 2. Layer in the neural networks
As for the input layer and hidden layers, we used the GCN layers’ update function to update each layer:
![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/GCN_Layer.png)

Because every neural network layer can be written as a non-linear function. The A denotes the adjacent matrix of the graph. So, we can get the l+1 layer’s neural network by l layer and the adjacency matrix. 
(Reference: https://tkipf.github.io/graph-convolutional-networks/)

![image](https://github.com/dongfang98/Self-supervised_RCD_System/blob/main/Picture/NeuralNetwork.png)
