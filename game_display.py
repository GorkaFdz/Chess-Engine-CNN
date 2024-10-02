# importing required librarys
import pygame
import chess
import math
import chess
import numpy as np
from tensorflow.keras.models import load_model
import threading
import chess.polyglot

#initialise display
X = 800
Y = 800
scrn = pygame.display.set_mode((X, Y))
pygame.init()

#basic colours
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)

#initialise chess board
b = chess.Board()

#load piece images
pieces = {'p': pygame.image.load('images/b_pawn.png').convert_alpha(),
          'n': pygame.image.load('images/b_knight.png').convert_alpha(),
          'b': pygame.image.load('images/b_bishop.png').convert_alpha(),
          'r': pygame.image.load('images/b_rook.png').convert_alpha(),
          'q': pygame.image.load('images/b_queen.png').convert_alpha(),
          'k': pygame.image.load('images/b_king.png').convert_alpha(),
          'P': pygame.image.load('images/w_pawn.png').convert_alpha(),
          'N': pygame.image.load('images/w_knight.png').convert_alpha(),
          'B': pygame.image.load('images/w_bishop.png').convert_alpha(),
          'R': pygame.image.load('images/w_rook.png').convert_alpha(),
          'Q': pygame.image.load('images/w_queen.png').convert_alpha(),
          'K': pygame.image.load('images/w_king.png').convert_alpha(),
          }

def update(scrn, board):
    '''
    Updates the screen based on the board class, colors the squares, and draws pieces.
    '''
    # Define colors
    WHITE = (247, 239, 210)
    BLACK = (105, 98, 76)

    # Draw the chessboard
    square_size = 100  # Since the screen is 800x800 and 8x8 grid, each square is 100x100

    for row in range(8):  # 8 rows
        for col in range(8):  # 8 columns
            # Calculate position of each square
            x = col * square_size
            y = (7 - row) * square_size  # Flip vertically to align with how chess board indices work

            # Choose color based on position (alternate between black and white)
            if (row + col) % 2 == 0:
                pygame.draw.rect(scrn, BLACK, (x, y, square_size, square_size))
            else:
                pygame.draw.rect(scrn, WHITE, (x, y, square_size, square_size))

    # Draw the pieces on the board
    for i in range(64):
        piece = board.piece_at(i)
        if piece is None:
            continue
        else:
            # Get the rectangle of the piece image to get its size
            piece_rect = pieces[str(piece)].get_rect()

            # Calculate the position to center the piece within the square
            x_position = (i % 8) * square_size + (square_size - piece_rect.width) // 2
            y_position = (7 - (i // 8)) * square_size + (square_size - piece_rect.height) // 2

            # Blit (draw) the piece at the calculated centered position
            scrn.blit(pieces[str(piece)], (x_position, y_position))

    # Update the display
    # pygame.display.flip()


def board_to_tensor(board):
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

def evaluate_position(model, board):
    tensor = board_to_tensor(board)
    evaluation = model.predict(np.array([tensor]), verbose=0)[0][0]
    return evaluation

def get_best_move(model, board, depth=3, alpha=float('-inf'), beta=float('inf'), ai_turn=True):
    if depth == 0 or board.is_game_over():
        return None, evaluate_position(model, board)
    
    with chess.polyglot.open_reader("codekiddy.bin") as reader:
        try:
            best_move = reader.weighted_choice(board).move
            return best_move, 0.0
        except IndexError:
            pass

    zobrist_hash = chess.polyglot.zobrist_hash(board)
    if zobrist_hash in transposition_table:
        stored_depth, stored_move, stored_eval, stored_flag = transposition_table[zobrist_hash]
        if stored_depth >= depth:
            if stored_flag == 'EXACT':
                return stored_move, stored_eval
            elif stored_flag == 'LOWERBOUND':
                alpha = max(alpha, stored_eval)
            elif stored_flag == 'UPPERBOUND':
                beta = min(beta, stored_eval)
            if alpha >= beta:
                return stored_move, stored_eval

    best_move = None
    if board.turn == chess.WHITE:
        max_eval = float('-inf')
        for move in sorted_moves(board):
            board.push(move)
            if board.is_checkmate() and ai_turn:
                board.pop()
                return move, float('inf')
            _, eval = get_best_move(model, board, depth-1, alpha, beta, not ai_turn)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        flag = 'EXACT' if alpha < max_eval < beta else ('LOWERBOUND' if max_eval >= beta else 'UPPERBOUND')
        transposition_table[zobrist_hash] = (depth, best_move, max_eval, flag)

        return best_move, max_eval
    else:
        min_eval = float('inf')
        for move in sorted_moves(board):
            board.push(move)
            if board.is_checkmate() and ai_turn:
                board.pop()
                return move, float('-inf')
            
            _, eval = get_best_move(model, board, depth-1, alpha, beta, not ai_turn)
            board.pop()
            
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break

        flag = 'EXACT' if alpha < min_eval < beta else ('UPPERBOUND' if min_eval <= alpha else 'LOWERBOUND')
        transposition_table[zobrist_hash] = (depth, best_move, min_eval, flag)
        
        return best_move, min_eval

def sorted_moves(board):
    # Move ordering: checks, captures, promotions, then other moves
    return sorted(board.legal_moves, 
                  key=lambda move: (board.is_check(),
                                    board.is_capture(move),
                                    move.promotion is not None,
                                    board.gives_check(move)),
                  reverse=True)

def main(BOARD, agent, agent_color):

    # Name the window
    pygame.display.set_caption('Chess')

    # Make background black
    scrn.fill(BLACK)

    # Variable to be used later
    index_moves = []

    status = True
    ai_thinking = False
    ai_move = None  # This will hold the AI move once it's calculated
    def ai_calculate_move():
        nonlocal ai_move
        ai_move, _ = agent(model, BOARD, depth=1)
    while status:
        # Clear the screen before each frame
        scrn.fill(BLACK)

        # Update the screen to draw the board and pieces
        update(scrn, BOARD)

        if BOARD.turn == agent_color and not ai_thinking:
            # Start the AI move calculation in a separate thread
            ai_thinking = True
            threading.Thread(target=ai_calculate_move).start()

        if ai_move is not None:
            # Once the AI move is ready, push it to the board
            BOARD.push(ai_move)
            ai_move = None  # Reset after move is made
            ai_thinking = False  # AI finished thinking
            # update(scrn, BOARD)

        else:
            for event in pygame.event.get():
                # Handle quitting
                if event.type == pygame.QUIT:
                    status = False

                # Handle mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the mouse click position
                    pos = pygame.mouse.get_pos()

                    # Calculate which square was clicked
                    square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                    index = (7 - square[1]) * 8 + square[0]

                    # If the player is making a move
                    if index in index_moves:
                        move = moves[index_moves.index(index)]
                        BOARD.push(move)
                        index_moves = []  # Reset after the move is made
                        update(scrn, BOARD)
                    else:
                        # Show possible moves for the selected piece
                        piece = BOARD.piece_at(index)
                        if piece is not None:
                            all_moves = list(BOARD.legal_moves)
                            moves = []

                            # Collect possible moves and highlight the squares
                            for m in all_moves:
                                if m.from_square == index:
                                    moves.append(m)
                                    t = m.to_square
                                    TX1 = 100 * (t % 8)
                                    TY1 = 100 * (7 - t // 8)

                                    # Highlight the possible moves with a blue rectangle
                                    pygame.draw.rect(scrn, BLUE, pygame.Rect(TX1, TY1, 100, 100),5)

                            # Update the display to show the highlights
                            pygame.display.flip()

                            # Store possible move indices for making the actual move
                            index_moves = [m.to_square for m in moves]
        for move in index_moves:
            TX1 = 100 * (move % 8)
            TY1 = 100 * (7 - move // 8)
            pygame.draw.rect(scrn, BLUE, pygame.Rect(TX1, TY1, 100, 100),5)

        # Update the display after everything is drawn
        pygame.display.flip()
        # If the game has ended, exit the loop
        if BOARD.outcome() is not None and not ai_thinking:
            print("asdfas")
            print(BOARD.outcome())
            status = False
            print(BOARD)

    pygame.quit()


model = load_model("engine_db.keras")
transposition_table = {}
main(b,get_best_move,False)
