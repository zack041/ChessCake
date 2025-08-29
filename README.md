# ChessCake
A Python-based neural network chess engine with an estimated Elo of 2200–2300 (based on games played against Chess.com bots). Uses a default search time of 5 seconds per move.
The CNN variant plays at an estimated Elo of 1600-1700

![logo](https://github.com/zack041/ChessCake/blob/main/docs/logo.jpg)

# Preview

Evaluation is measured in centipawns (1/100 of a pawn value), while search depth is measured in plies (half-moves).

The following output demonstrates the program’s execution:

r n b q k b n r <br>
p p p p p p p p <br>
. . . . . . . . <br>
. . . . . . . . <br>
. . . . . . . . <br>
. . . . . . . . <br>
P P P P P P P P <br>
R N B Q K B N R <br>
search depth 10 <br>
depth reached: 7 <br>
evaluation: 15.1598482131958 <br>
best move: g1f3 <br>
nodes searched: 33480 <br>
time to search: 5.000157833099365 <br>
 <br>
make move g1f3 <br>
r n b q k b n r <br>
p p p p p p p p <br>
. . . . . . . . <br>
. . . . . . . . <br>
. . . . . . . . <br>
. . . . . N . . <br>
P P P P P P P P <br>
R N B Q K B . R <br>
search depth 10 <br>
depth reached: 7 <br>
evaluation: 9.403926849365234 <br>
best move: g8f6 <br>
nodes searched: 46272 <br>
time to search: 5.000718116760254 <br>

# Features
1. Alpha-beta pruning
2. Transposition table
3. Move-ordering
4. Null move reductions
5. Killer heuristic
6. Late move reductions
7. Quiescence search
8. Iterative deepening
9. Neural networks

# Neural Network

Applies a compact 3-layer dense neural network with tanh activations. The model is trained using code provided in NNeval.py, and the training dataset originates from the Kaggle competition [Train Your Own Stockfish NNUE](https://www.kaggle.com/competitions/train-your-own-stockfish-nnue/overview). The Keras model is converted to ONNX format using the script keras2onnx.py, to be used in the engine.

The CNN variant consists of three convolutional layers(32, 64, 128 filters) followed by a 256-unit dense layer and a final 1-unit output.
