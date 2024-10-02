# Chess-Engine-CNN

This project is a Convolutional Neural Network trained in python to evaluate chess positions. It can be trained using the `engine.py` script. The data used for training was downloaded from ![lichess database](https://database.lichess.org/) from January 2013. It used a total of 30,000 games from the database and taking every unique position from those games the network was trained with 1,765,327 positions. Each chess position was transformed into a 12x8x8 NumPy array, where each 8x8 matrix corresponds to a specific type of piece (e.g., pawns, knights, etc.), and a value of 1 indicates the presence of a piece on a particular square. Separate matrices are used for white and black pieces. These positions were evaluated by a local version of Stockfish in order to feed the model with a position and corresponding evaluation, and later to test the network.

## Installation

1. Ensure you have a working version of Stockfish installed on your machine. The default path used in the project is `/usr/games/stockfish`.
2. Clone this repository.
3. Install dependencies using `pip` (tested using Python 3.10.12): `pip install -r requirements.txt`.

## Evaluating the engine

If you want to test the engine for different positions, it can be done modifying the `evaluation.py` file. Here add the desired chess positions as FEN strings to the *test_positions* list at the end of the script and then run it to see the board, Stockfish evaluation and the predicted evaluation next to each other.

## Playing the engine

You can play against engine by directly running the `game_display.py` script. This is a simple pygame game which shows the board and can be played by clicking on the pieces. The code is a modified version of the implementation found in ![this article](https://medium.com/dev-genius/simple-interactive-chess-gui-in-python-c6d6569f7b6c). In order to play a move, the AI finds the best move using a negamax algorithm and the evaluations from the network except for the openings for which it uses an opening book. The model's search depth for moves can be adjusted, with a default depth of 3.

## Testing the engine on lichess

I created a bot account, named ![grkbot](https://lichess.org/@/grkbot), for the engine using ![the official repository from lichess](https://github.com/lichess-bot-devs/lichess-bot). The engine performed well against bots rated around 1000 Elo but struggled against stronger opponents. Since the time required to compute each move increases exponentially with depth, the search depth was set to 3, resulting in an average of 90 seconds per move. As a result, the engine primarily played classical games (30+15 time control).

## Conclusion

Overall, I am pleased with how the project turned out. The engine performs well at lower levels and showcases the potential of using a Convolutional Neural Network to evaluate chess positions. However, there is still room for improvement, particularly in optimizing the model's speed and improving its performance against higher-rated engines. Future enhancements could involve refining the evaluation function and experimenting with different data preprocessing techniques, such as incorporating information about controlled squares or attacked pieces for each side. Despite these opportunities for growth, I am happy with the current state of the engine and its ability to compete at the 1000 Elo level on Lichess.
