Deterministic graph algorithm for node embedding generation based on their closeness to each other

This unsupervised learning technique is based on the idea of "signal" transfer from 
one node to the rest of the graph and is also inspired by Google's PageRank algorithm.
Main use-case of the algorithm so far is social networks clustering and community detection.
Please refer to the jupyter notebook in this repository for the exampples.


# The idea
Here's presented the basic idea for directed/undirected unweighted graph.

1. Every node is assigned a vector. For node i the vector's j-th element is a number representing its closeness to node j (you can think of it as signal strength).
2. The closeness of the node i to the node j (the "central" node the signal flows from) is defined as a sum of the signal strengths of the neighboring nodes multiplied by the damping factor parameter. (The dumping factor works like a kind of distance penalty)
3. Signal a node "emits" to other connected nodes equal to the given signal's strength in the node itself divided by the number of edges it is able to emit to.
4. Initially, the node j is assigned a closeness (to itself) equal to 1.


# The implementation
The implementation presented in this repository gets things done through solving systems of equations.
For every node *j* its signal strength in all other nodes is computed through a single system of linear equations with *N* equations and *N* unknown variables, where *N* is the number of nodes in the graph.
Those equations are assumed to be sparse, so the systems are stored as a CSR matrices since it drastically reduces the space complexity.
This implementation relies on [pypardiso][1] anaconda package for solving sparse linear equations.

This implementation provides classes to work with Networkx's directed and undirected graphs. 
Please note, in this implementation node *i* can receive a signal from node *j* only if there's an edge *i-j*, in other words, direction of the signal flow is opposite to the direction of the edge, resembling Instagram's "follow" relation.

[1]: https://github.com/haasad/PyPardisoProject

# Example
Here's an example of application of the algorithm to the famous Zachary's karate club dataset.

Zachary's karate club split:

![Test split](https://github.com/monomonedula/simple-graph-embedding/blob/master/zachary_expected.png "Zachary's karate club split")

Zachary's karate club split predicted with K-Means clustering on the generated embeddings with damping factor of 0.7:

![Generated split](https://github.com/monomonedula/simple-graph-embedding/blob/master/zachary_computed.png "Zachary's karate club split predicted with K-Means clustering on the generated embeddings with damping factor of 0.7")
