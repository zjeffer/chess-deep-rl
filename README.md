# Chess engine using Deep Reinforcement learning

To run one MCTS simulation:

1. To traverse the tree, select the edge with max Q+U value
	* Q = mean value of the state over all simulations in this search
	* U = upper confidence bound
	* Do this until a leaf node is reached
2. Expand the leaf node. Use the neural network to select the best move. This will generate a new state
	* If the new state is a terminal state, stop here
	* If the new state is not terminal, keep expanding until we reach a terminal state (=rollout)
	* N = 0, W = 0, Q = 0, P = p_a (probability for that action)
3. Backpropagation: update the values from each node visited in step 2
	* N = N + 1
	* W = W + v
	* Q = W / N



To train the network, optimize the weights of the neural network by comparing the neural network's predictions to the actual game outcome.

## Useful sources

### Wikipedia articles & Library documentation

* https://en.wikipedia.org/wiki/Deep_reinforcement_learning
* https://en.wikipedia.org/wiki/Reinforcement_learning
* https://en.wikipedia.org/wiki/AlphaZero
* https://en.wikipedia.org/wiki/AlphaGo & https://en.wikipedia.org/wiki/AlphaGo_Zero
* https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
* https://en.wikipedia.org/wiki/Minimax What stockfish uses
* https://en.wikipedia.org/wiki/Alpha-beta_pruning What stockfish uses
* https://python-chess.readthedocs.io/en/latest/ Python chess library
* https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero  LC0's technical explanation


### AlphaZero & AlphaGo Zero specific articles & papers

* https://arxiv.org/abs/1712.01815 The AlphaZero paper
* https://www.science.org/doi/10.1126/science.aar6404 Supplementary materials for the paper: more info
* http://web.stanford.edu/~surag/posts/alphazero.html
* https://www.sciencedirect.com/science/article/pii/S0925231221005245
* https://chess.stackexchange.com/questions/19353/understanding-alphazero
* https://chess.stackexchange.com/questions/19401/how-does-alphazero-learn-to-evaluate-a-position-it-has-never-seen

* https://towardsdatascience.com/can-deep-reinforcement-learning-solve-chess-b9f52855cd1e

* https://www.nature.com/articles/nature24270/figures/2 From AlphaGo Zero paper: MCTS 

* https://link.springer.com/chapter/10.1007/3-540-45579-5_18

* https://joshvarty.github.io/AlphaZero/ and https://github.com/JoshVarty/AlphaZeroSimple
* https://chess.stackexchange.com/a/37477 Explanation for input of neural network

### Diagrams

* https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0  Useful diagram for AlphaGo Zero


### Tutorials

* https://towardsdatascience.com/alphazero-a-novel-reinforcement-learning-algorithm-deployed-in-javascript-56018503ad18 More info about the algorithm
* https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
* https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a


## Interesting videos

* https://www.youtube.com/watch?v=uPUEq8d73JI Lex Fridman + David Silver
* https://www.youtube.com/watch?v=2pWv7GOvuf0 Lecture RL from David Silver
* https://www.youtube.com/watch?v=A3ekFcZ3KNw: Keynote David Silver NIPS 2017