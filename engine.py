import json
from dotenv import load_dotenv
import numpy as np
import chess
import chess.pgn
import io
import chess.engine
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import multiprocessing as mp


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    print(f"Test Set Evaluation:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")


def parse_games(str):
    json_objects = [obj for obj in str.strip().split('\n') if obj]
    return [json.loads(obj) for obj in json_objects]


def board_to_tensor(board):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((12, 8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_index = piece_map[piece.symbol()]
            row, col = divmod(square, 8)
            tensor[piece_index, row, col] = 1
    return tensor


def stockfish_evaluation(board, engine, time_limit=0.1, depth_limit=10):
    try:
        result = engine.analyse(board, chess.engine.Limit(time=time_limit, depth=depth_limit))
        return result['score']
    except Exception as e:
        print(f"Error in stockfish_evaluation: {e}")
        return None


def process_game(pgn_string):
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    board = game.board()
    x = []
    y = []
    try:
        for move in game.mainline_moves():
            board.push(move)
            eval = stockfish_evaluation(board, engine)
            if eval is not None:
                score = eval.white().score(mate_score=5000) / 100
                x.append(board_to_tensor(board))
                y.append(score)
    except Exception as e:
        print(f"Error processing game: {e}")
    finally:
        engine.quit()
        print(f"Processed {len(x)} moves")
    return x, y


def process_games_parallel(pgn_strings):
    with mp.Pool() as pool:
        results = pool.map(process_game, pgn_strings)
    x = []
    y = []
    for x_game, y_game in results:
        x.extend(x_game)
        y.extend(y_game)
    return x, y


def main():
    all_pgn_strings = []
    with open("lichess_db_standard_rated_2013-01.pgn") as pgn_file:
        game_count = 0
        # We evalueate 30.000 games
        n = 30000
        while game_count < n:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            string_io = io.StringIO()
            exporter = chess.pgn.FileExporter(string_io)
            game.accept(exporter)
            all_pgn_strings.append(string_io.getvalue())
            game_count += 1
            print(f"Completed: {game_count/n*100:.2f}%")

    print("Starting parallel processing...")
    x, y = process_games_parallel(all_pgn_strings)
    print("Parallel processing completed.")
    print(f"Total positions processed: {len(x)}")
    # Save the lists to a file using pickle
    x = np.array(x)
    y = np.array(y)
    np.savez_compressed("x", x)
    np.savez_compressed("y", y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    tf.keras.backend.clear_session()
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(12, 8, 8)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=100,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae}")
    print(f"Test loss: {test_loss}")

    model.save("engine_db.keras")

    print("\nEvaluating model on test set:")
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
