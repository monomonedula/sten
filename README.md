Experimental graph algorithm for node embedding generation 

This unsupervised learning tecnique is based on the idea of "signal" transfer from 
one node to the rest of the graph and is also inspired by Google's PageRank algorithm.
Main use-case of the algorithm so far is social networks clustering and community detection.
Please refer to the jupyter notebook in this repository for the exampples.


# The idea
1. Every node is assigned a vector. For node *i* the vector's *j*-th element is a number representing its closeness to node *j* (you can think of it as signal strength).
2. The closeness of the node *i* to the node *j* (the "central" node the singal flows from) 
is defined as a sum of the signal strengths of the neighboring nodes.
3. Initially, the node *j* is assigned a closeness (to itself) equal to 1.

# The implementation
The implementation presented in this repository gets things done through solving systems of equations.
For every node *j* its signal strength in all other nodes is computed through a single system of linear equations with *N* equations and *N* unknown variables, where *N* is the number of nodes in the graph.
Those equations are assumed to be sparse, so the systems are stored as a CSR matrices since it drastically reduces the space complexity.
This implementation relies on [pypardiso][1] anaconda package for solving sparse linear equations.

This implementation provides classes to work with Networkx's directed and undirected graphs. 
Please note, in this implementation node *i* can receive a signal from node *j* only if there's an edge *i-j*, in other words, direction of the signal flow is opposite to the direction of the edge, resembling Instagram's "follow" relation.

[1]: https://github.com/haasad/PyPardisoProject
