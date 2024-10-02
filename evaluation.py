import chess
import chess.svg
import chess.engine
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

class ChessPositionEvaluator:
    def __init__(self, model_path, engine_path):
        self.board = chess.Board()
        self.model = load_model(model_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def load_model(self, model_path):
        # Load your model here
        # return ChessEvaluationModel.load(model_path)
        pass

    def set_position(self, fen):
        self.board.set_fen(fen)

    def board_to_tensor(self, board):
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_index = piece_map[piece.symbol()]
                row, col = divmod(square, 8)
                tensor[piece_index, row, col] = 1.0
        return tensor

    def evaluate_position(self, board):
        tensor = self.board_to_tensor(board)
        evaluation = self.model.predict(np.array([tensor]), verbose=0)[0][0]
        return evaluation

    def stockfish_evaluation(self, board, time_limit=0.1, depth_limit=10):
        try:
            result = self.engine.analyse(board, chess.engine.Limit(time=time_limit, depth=depth_limit))
            return result['score'].white().score() / 100  # Convert centipawns to pawns
        except Exception as e:
            print(f"Error in stockfish_evaluation: {e}")
            return None

    def get_actual_evaluation(self):
        return self.stockfish_evaluation(self.board)

    def get_predicted_evaluation(self):
        return self.evaluate_position(self.board)

    def visualize_position(self):
        svg = chess.svg.board(self.board, size=400)
        png = self.svg_to_png(svg)
        return Image.open(BytesIO(png))

    @staticmethod
    def svg_to_png(svg_string):
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_string)

    def evaluate_and_visualize(self, fen):
        self.set_position(fen)
        actual_eval = self.get_actual_evaluation()
        predicted_eval = self.get_predicted_evaluation()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot chess board
        chess_img = self.visualize_position()
        ax1.imshow(chess_img)
        ax1.axis('off')
        ax1.set_title('Chess Position')

        # Plot evaluation comparison
        labels = ['Actual (Stockfish)', 'Predicted (Neural Network)']
        evaluations = [actual_eval, predicted_eval]
        ax2.bar(labels, evaluations)
        ax2.set_ylabel('Evaluation (pawns)')
        ax2.set_title('Evaluation Comparison')
        ax2.axhline(y=0, color='r', linestyle='-', linewidth=0.5)

        for i, v in enumerate(evaluations):
            ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')

        plt.tight_layout()
        plt.show()

        print(f"FEN: {fen}")
        print(f"Actual Evaluation (Stockfish): {actual_eval:.2f}")
        print(f"Predicted Evaluation (Neural Network): {predicted_eval:.2f}")
        print(f"Difference: {actual_eval - predicted_eval:.2f}")

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()

# Usage example
model_path = "engine_db.keras"
engine_path = "/usr/games/stockfish"
evaluator = ChessPositionEvaluator(model_path, engine_path)

test_positions = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
    "rnbqkb1r/ppp1pppp/5n2/3P4/8/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "rnbqkb1r/ppp1pppp/8/3n4/8/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 4",
    "rnbqkb1r/ppp1pppp/8/3n4/8/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 1 4",
    "rn1qkb1r/ppp1pppp/8/3n4/6b1/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 2 5",
    "rn1qkb1r/ppp1pppp/8/3n4/6b1/3P1N1P/PPP2PP1/RNBQKB1R b KQkq - 0 5",
    "rn1qkb1r/ppp1pppp/8/3n3b/8/3P1N1P/PPP2PP1/RNBQKB1R w KQkq - 1 6",
    "rn1qkb1r/ppp1pppp/8/3n3b/8/2NP1N1P/PPP2PP1/R1BQKB1R b KQkq - 2 6",
    "rn1qkb1r/ppp1pppp/8/7b/8/2nP1N1P/PPP2PP1/R1BQKB1R w KQkq - 0 7",
    "rn1qkb1r/ppp1pppp/8/7b/8/2PP1N1P/P1P2PP1/R1BQKB1R b KQkq - 0 7",
    "rn1qkb1r/ppp2ppp/4p3/7b/8/2PP1N1P/P1P2PP1/R1BQKB1R w KQkq - 0 8",
    "rn1qkb1r/ppp2ppp/4p3/6Bb/8/2PP1N1P/P1P2PP1/R2QKB1R b KQkq - 1 8",
    "rn1qkb1r/ppp3pp/4pp2/6Bb/8/2PP1N1P/P1P2PP1/R2QKB1R w KQkq - 0 9",
    "rn1qkb1r/ppp3pp/4pp2/7b/8/2PPBN1P/P1P2PP1/R2QKB1R b KQkq - 1 9",
    "r2qkb1r/ppp3pp/2n1pp2/7b/8/2PPBN1P/P1P2PP1/R2QKB1R w KQkq - 2 10",
    "r2qkb1r/ppp3pp/2n1pp2/7b/8/2PPBN1P/P1P2PP1/RQ2KB1R b KQkq - 3 10",
    "r2qkb1r/p1p3pp/1pn1pp2/7b/8/2PPBN1P/P1P2PP1/RQ2KB1R w KQkq - 0 11",
    "r2qkb1r/p1p3pp/1pn1pp2/1Q5b/8/2PPBN1P/P1P2PP1/R3KB1R b KQkq - 1 11",
    "r2qkb1r/p1p3pp/1pn1pp2/1Q6/8/2PPBb1P/P1P2PP1/R3KB1R w KQkq - 0 12",
    "r2qkb1r/p1p3pp/1pn1pp2/1Q6/8/2PPBP1P/P1P2P2/R3KB1R b KQkq - 0 12",
    "r3kb1r/p1pq2pp/1pn1pp2/1Q6/8/2PPBP1P/P1P2P2/R3KB1R w KQkq - 1 13",
    "r3kb1r/p1pq2pp/1pn1pp2/1Q6/8/2PPBP1P/P1P2PB1/R3K2R b KQkq - 2 13",
    "r3k2r/p1pq2pp/1pnbpp2/1Q6/8/2PPBP1P/P1P2PB1/R3K2R w KQkq - 3 14",
    "r3k2r/p1pq2pp/1pnbpp2/1Q6/5P2/2PPB2P/P1P2PB1/R3K2R b KQkq - 0 14",
    "r3k2r/p1pq2pp/1p1bpp2/nQ6/5P2/2PPB2P/P1P2PB1/R3K2R w KQkq - 1 15",
    "r3k2r/p1pQ2pp/1p1bpp2/n7/5P2/2PPB2P/P1P2PB1/R3K2R b KQkq - 0 15",
    "r6r/p1pk2pp/1p1bpp2/n7/5P2/2PPB2P/P1P2PB1/R3K2R w KQ - 0 16",
    "B6r/p1pk2pp/1p1bpp2/n7/5P2/2PPB2P/P1P2P2/R3K2R b KQ - 0 16",
    "r7/p1pk2pp/1p1bpp2/n7/5P2/2PPB2P/P1P2P2/R3K2R w KQ - 0 17",
    "r7/p1pk2pp/1p1bpp2/n7/5P2/2PPB2P/P1P2P2/R4RK1 b - - 1 17",
    "r7/p1pk2pp/1p1b1p2/n3p3/5P2/2PPB2P/P1P2P2/R4RK1 w - - 0 18",
    "r7/p1pk2pp/1p1b1p2/n3pP2/8/2PPB2P/P1P2P2/R4RK1 b - - 0 18",
    "r7/p1pk2pp/1pnb1p2/4pP2/8/2PPB2P/P1P2P2/R4RK1 w - - 1 19",
    "r7/p1pk2pp/1pnb1p2/4pP2/P7/2PPB2P/2P2P2/R4RK1 b - - 0 19",
    "r7/p1pkn1pp/1p1b1p2/4pP2/P7/2PPB2P/2P2P2/R4RK1 w - - 1 20",
    "r7/p1pkn1pp/1p1b1p2/P3pP2/8/2PPB2P/2P2P2/R4RK1 b - - 0 20",
    "r7/p1pk2pp/1p1b1p2/P3pn2/8/2PPB2P/2P2P2/R4RK1 w - - 0 21",
    "r7/p1pk2pp/1P1b1p2/4pn2/8/2PPB2P/2P2P2/R4RK1 b - - 0 21",
    "r7/p1pk2pp/1P1b1p2/4p3/8/2PPn2P/2P2P2/R4RK1 w - - 0 22",
    "r7/p1pk2pp/1P1b1p2/4p3/8/2PPP2P/2P5/R4RK1 b - - 0 22",
    "r7/p2k2pp/1p1b1p2/4p3/8/2PPP2P/2P5/R4RK1 w - - 0 23",
    "r7/p2k2pp/1p1b1p2/4p3/8/2PPP2P/2P5/R2R2K1 b - - 1 23",
    "r7/p5pp/1pkb1p2/4p3/8/2PPP2P/2P5/R2R2K1 w - - 2 24",
    "r7/p5pp/1pkb1p2/4p3/R7/2PPP2P/2P5/3R2K1 b - - 3 24",
    "r7/p5p1/1pkb1p2/4p2p/R7/2PPP2P/2P5/3R2K1 w - - 0 25",
    "r7/p5p1/1pkb1p2/4p2p/R7/2PPP2P/2P5/R5K1 b - - 1 25",
    "r7/6p1/1pkb1p2/p3p2p/R7/2PPP2P/2P5/R5K1 w - - 0 26",
    "r7/6p1/1pkb1p2/p3p2p/2R5/2PPP2P/2P5/R5K1 b - - 1 26",
    "r7/1k4p1/1p1b1p2/p3p2p/2R5/2PPP2P/2P5/R5K1 w - - 2 27",
    "r7/1k4p1/1p1b1p2/p3p2p/2R5/2PPP2P/2P5/R6K b - - 3 27",
    "r7/1k6/1p1b1p2/p3p1pp/2R5/2PPP2P/2P5/R6K w - - 0 28",
    "r7/1k6/1p1b1p2/p3p1pp/2R4P/2PPP3/2P5/R6K b - - 0 28",
    "r7/1k6/1p1b1p2/p3p2p/2R3pP/2PPP3/2P5/R6K w - - 0 29",
    "r7/1k6/1p1b1p2/p3p2p/2R3pP/2PPP3/2P5/5R1K b - - 1 29",
    "5r2/1k6/1p1b1p2/p3p2p/2R3pP/2PPP3/2P5/5R1K w - - 2 30",
    "5r2/1k6/1p1b1p2/p3pR1p/2R3pP/2PPP3/2P5/7K b - - 3 30",
    "5r2/1k6/3b1p2/pp2pR1p/2R3pP/2PPP3/2P5/7K w - - 0 31",
    "5r2/1k6/3b1p2/pp2pR1p/4R1pP/2PPP3/2P5/7K b - - 1 31",
    "5r2/1k6/3b1p2/1p2pR1p/p3R1pP/2PPP3/2P5/7K w - - 0 32",
    "5r2/1k6/3b1p2/1p2p2R/p3R1pP/2PPP3/2P5/7K b - - 0 32",
    "5r2/1k6/3b1p2/1p2p2R/4R1pP/p1PPP3/2P5/7K w - - 0 33",
    "5r2/1k6/3b1p2/1p2p2R/6RP/p1PPP3/2P5/7K b - - 0 33",
    "5r2/1k6/3b1p2/1p2p2R/6RP/2PPP3/p1P5/7K w - - 0 34",
    "5r2/1k6/3b1p2/1p2p2R/7P/2PPP3/p1P5/6RK b - - 1 34",
    "r7/1k6/3b1p2/1p2p2R/7P/2PPP3/p1P5/6RK w - - 2 35",
    "r7/1k6/3b1p2/1p2p2R/7P/2PPP3/p1P5/R6K b - - 3 35",
    "r7/1k6/3b1p2/4p2R/1p5P/2PPP3/p1P5/R6K w - - 0 36",
    "r7/1k5R/3b1p2/4p3/1p5P/2PPP3/p1P5/R6K b - - 1 36",
    "r7/7R/2kb1p2/4p3/1p5P/2PPP3/p1P5/R6K w - - 2 37",
    "r7/7R/2kb1p2/4p3/1pP4P/3PP3/p1P5/R6K b - - 0 37",
    "r7/7R/2k2p2/2b1p3/1pP4P/3PP3/p1P5/R6K w - - 1 38",
    "r7/5R2/2k2p2/2b1p3/1pP4P/3PP3/p1P5/R6K b - - 2 38",
    "r7/5R2/2k2p2/4p3/1pP4P/3Pb3/p1P5/R6K w - - 0 39",
    "r7/8/2k2R2/4p3/1pP4P/3Pb3/p1P5/R6K b - - 0 39",
    "r7/8/5R2/2k1p3/1pP4P/3Pb3/p1P5/R6K w - - 1 40",
    "r7/8/5R2/2k1p2P/1pP5/3Pb3/p1P5/R6K b - - 0 40",
    "r7/8/5R2/2k1p2P/1pPb4/3P4/p1P5/R6K w - - 1 41",
    "r7/8/5R2/2k1p2P/1pPb4/3P4/R1P5/7K b - - 0 41",
    "8/8/5R2/2k1p2P/1pPb4/3P4/r1P5/7K w - - 0 42",
    "8/8/5R1P/2k1p3/1pPb4/3P4/r1P5/7K b - - 0 42",
    "r7/8/5R1P/2k1p3/1pPb4/3P4/2P5/7K w - - 1 43",
    "r7/8/7P/2k1pR2/1pPb4/3P4/2P5/7K b - - 2 43",
    "7r/8/7P/2k1pR2/1pPb4/3P4/2P5/7K w - - 3 44",
    "7r/8/7P/2k1p2R/1pPb4/3P4/2P5/7K b - - 4 44",
    "7r/8/7P/2k1p2R/1pP5/2bP4/2P5/7K w - - 5 45",
    "7r/8/7P/2k1p2R/1pP5/2bP4/2P3K1/8 b - - 6 45",
    "7r/8/7P/4p2R/1pPk4/2bP4/2P3K1/8 w - - 7 46",
    "7r/8/7P/4p2R/1pPk4/2bP1K2/2P5/8 b - - 8 46",
    "7r/8/7P/4p2R/1pPk4/3P1K2/2Pb4/8 w - - 9 47",
    "7r/8/7P/4p3/1pPk3R/3P1K2/2Pb4/8 b - - 10 47",
    "7r/8/7P/2k1p3/1pP4R/3P1K2/2Pb4/8 w - - 11 48",
    "7r/7P/8/2k1p3/1pP4R/3P1K2/2Pb4/8 b - - 0 48",
    "7r/7P/8/2k1p3/1pP4R/2bP1K2/2P5/8 w - - 1 49",
    "7r/7P/8/2k1p2R/1pP5/2bP1K2/2P5/8 b - - 2 49",
    "7r/7P/8/4p2R/1pPk4/2bP1K2/2P5/8 w - - 3 50",
    "7r/7P/8/4p3/1pPk3R/2bP1K2/2P5/8 b - - 4 50",
    "7r/7P/8/2k1p3/1pP4R/2bP1K2/2P5/8 w - - 5 51",
    "7r/7P/8/2k1p2R/1pP5/2bP1K2/2P5/8 b - - 6 51",
    "7r/7P/8/4p2R/1pPk4/2bP1K2/2P5/8 w - - 7 52",
    "7r/7P/8/4p3/1pPk3R/2bP1K2/2P5/8 b - - 8 52",
    "7r/7P/8/2k1p3/1pP4R/2bP1K2/2P5/8 w - - 9 53",
    "7r/7P/8/2k1p3/1pP1K2R/2bP4/2P5/8 b - - 10 53",
    "7r/7P/3k4/4p3/1pP1K2R/2bP4/2P5/8 w - - 11 54",
    "7r/7P/3k4/4pK2/1pP4R/2bP4/2P5/8 b - - 12 54",
    "5r2/7P/3k4/4pK2/1pP4R/2bP4/2P5/8 w - - 13 55",
    "5r2/7P/3k2K1/4p3/1pP4R/2bP4/2P5/8 b - - 14 55",
]

for fen in test_positions:
    evaluator.evaluate_and_visualize(fen)
