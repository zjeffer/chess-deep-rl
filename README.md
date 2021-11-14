# Chess engine using Deep Reinforcement learning

1. Use MCTS to choose the next move (PUCT = Predictor + Upper Confidence Bound tree search)
2. After a certain depth, use a neural network to evaluate the best moves found by MCTS
3. Select the best move from the neural network
4. Repeat steps 1-3 until the game is over

To train the network, optimize the weights of the neural network by comparing the neural network's predictions to the actual game outcome.

## Useful sources

* https://en.wikipedia.org/wiki/Deep_reinforcement_learning
* https://en.wikipedia.org/wiki/Reinforcement_learning
* https://en.wikipedia.org/wiki/AlphaZero
* https://en.wikipedia.org/wiki/AlphaGo & https://en.wikipedia.org/wiki/AlphaGo_Zero
* https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
* https://en.wikipedia.org/wiki/Alpha-beta_pruning
* https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
* https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero
* http://web.stanford.edu/~surag/posts/alphazero.html
* https://chess.stackexchange.com/questions/19353/understanding-alphazero
* https://chess.stackexchange.com/questions/19401/how-does-alphazero-learn-to-evaluate-a-position-it-has-never-seen
* https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
* https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a
* https://towardsdatascience.com/can-deep-reinforcement-learning-solve-chess-b9f52855cd1e
* https://arxiv.org/abs/1712.01815
* https://www.sciencedirect.com/science/article/pii/S0925231221005245
* https://link.springer.com/chapter/10.1007/3-540-45579-5_18
* https://python-chess.readthedocs.io/en/latest/

## Interesting videos

* https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
* https://www.youtube.com/watch?v=uPUEq8d73JI&t=317s