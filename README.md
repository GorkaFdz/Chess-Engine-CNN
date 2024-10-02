# Chess-Engine-CNN

This project is a Convolutional Neural Network trained in python to evaluate chess positions. It can be trained using the `engine.py` script. The data used for training was downloaded from ![lichess database](https://database.lichess.org/) from January 2013. It used a total of 30,000 games from the database and taking every unique position in those games the network was trained with 1,765,327 positions. These chess positions were converted into 12x8x8 numpy arrays where each 8x8 matrix represented the position of each piece with a 1 for the corresponding squares and 0 for the rest separated by white and black pieces. These positions were evaluated by a local version of Stockfish in order to feed the model with a position and corresponding evaluation and to later test the network.

### Installation

1. Clone this repository.
2. Install dependencies using pip (tested using Python 3.10.12): pip install -r requirements.txt
3. Ensure you have a working version of Stockfish installed on your machine. The default path used in the project is `/usr/games/stockfish`.

### Evaluating the engine
If you want to test the engine for different positions, it can be done modifying the `evaluation.py` file. Here add the desired chess positions as FEN strings to the test_positions list at the end of the script and then run it to see the board, Stockfish evaluation and the predicted evaluation next to each other.

### Playing the engine
It is possible to play the engine directly running the `game_display.py` script. This is a simple pygame game which shows the board and can be played by clicking on the pieces. The code I used is a modified version of ![this article](https://medium.com/dev-genius/simple-interactive-chess-gui-in-python-c6d6569f7b6c). In order to play a move, the AI finds the best move using a negamax algorithm and the evaluations from the network except for the openings for which it uses an opening book. The depth of moves the model can look at can be modified and is defaulted to 3.

### Testing the engine on lichess
I created a bot account for the engine using ![the official repository from lichess](https://github.com/lichess-bot-devs/lichess-bot). The engine was able to win games against some other bots rated around 1000 elo but did not perform that well into higher rated engines. Since the time required to find a move gets exponentially higher with each depth level, it was set to 3 with which it took an average of 90 seconds to find a move. For this reason the engine mainly played classical games (30+15).
