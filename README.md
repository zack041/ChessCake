# ChessCake
A Python-based neural network chess engine with an estimated Elo of 2200–2300, based on games played against Chess.com bots. Uses a default search time of 5 seconds per move.

![logo](https://github.com/zack041/ChessCake/blob/main/docs/logo.jpg)

# Preview

Evaluation is measured in centipawns (1/100 of a pawn value), while search depth is measured in plies (half-moves).

The following output demonstrates the program’s execution:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
search depth 10
depth reached: 7
evaluation: 15.1598482131958
best move: g1f3
nodes searched: 33480
time to search: 5.000157833099365
make move g1f3
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . N . .
P P P P P P P P
R N B Q K B . R
search depth 10
depth reached: 7
evaluation: 9.403926849365234
best move: g8f6
nodes searched: 46272
time to search: 5.000718116760254

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

Applies a compact 3-layer dense neural network with tanh activations. The model is trained using code provided in NNeval.py, and the training dataset originates from the Kaggle competition [Train Your Own Stockfish NNUE](https://www.kaggle.com/competitions/train-your-own-stockfish-nnue/overview). The Keras model is converted to ONNX format using the script keras2onnx.py, which is used in the engine.
