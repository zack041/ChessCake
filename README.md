# ChessCake
A Python-based neural network chess engine with an estimated Elo of 2200–2300, based on games played against Chess.com bots. Uses a default search time of 5 seconds per move.

![logo](https://github.com/zack041/ChessCake/blob/main/docs/logo.jpg)

# Features
1. Alpha-beta pruning
2. Transposition table
3. Move-ordering
4. Null move reductions
5. Killer heuristic
6. Late move reductions
7. Quiescence search
8. Iterative deepening
9. Neural network

# Neural Network

Applies a compact 3-layer dense neural network with tanh activations. The model is trained using code provided in NNeval.py, and the training dataset originates from the Kaggle competition [Train Your Own Stockfish NNUE](https://www.kaggle.com/competitions/train-your-own-stockfish-nnue/overview).
