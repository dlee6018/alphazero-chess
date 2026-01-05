import chess
import numpy as np
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "games.csv"

_df = None

def sample_board(batch_size: int = 1):
    """
    Samples a board from the chess dataset and returns the board states along with the winner.
    
    Returns:
        tuple: (boards, winner) where:
            - boards: list of chess.Board objects, one for each move made in the game
            - winner: int - 1 if white won, -1 if black won, 0 if tied
    """
    global _df
    if _df is None:
        _df = pd.read_csv(file_path)
    ex = _df.sample(batch_size)
    
    # Get the first row if batch_size > 1, otherwise just the row
    if batch_size == 1:
        row = ex.iloc[0]
    else:
        row = ex.iloc[0]  # For now, return first game if batch_size > 1
    
    # Parse moves string and reconstruct board
    moves_str = row['moves']
    moves_list = moves_str.split()
    
    game_board = chess.Board()
    boards = [game_board.copy()]  # Start with initial board state
    
    for move_str in moves_list:
        try:
            move = game_board.parse_san(move_str)
            game_board.push(move)
            boards.append(game_board.copy())
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            # If move parsing fails, try UCI format
            try:
                move = chess.Move.from_uci(move_str)
                game_board.push(move)
                boards.append(game_board.copy())
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                # Skip invalid moves
                continue
    
    # Determine winner: 1 for white, -1 for black, 0 for draw
    winner_str = row['winner'].lower()
    if winner_str == 'white':
        winner = 1
    elif winner_str == 'black':
        winner = -1
    else:  # 'draw' or any other status
        winner = 0
    
    return boards, winner


# for key,value in data.items():
  # print(key,value)
# Index(['id', 'rated', 'created_at', 'last_move_at', 'turns', 'victory_status',
#        'winner', 'increment_code', 'white_id', 'white_rating', 'black_id',
#        'black_rating', 'moves', 'opening_eco', 'opening_name', 'opening_ply'],
#       dtype='object')

def board_to_input_planes(board: chess.Board):
    """
    Returns a tensor of shape (18, 8, 8)
    suitable as input to an AlphaZero-like NN.
    """
    planes = []

    # ----------------------------------------------------
    # 1. Piece planes (12 planes)
    # ----------------------------------------------------
    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]

    # white & black
    for color in [chess.WHITE, chess.BLACK]:
        for piece in piece_order:
            plane = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece, color): # set of squares indices from 0-63
                rank = 7 - chess.square_rank(square)   # a,b,c, ..
                file = chess.square_file(square) # 1,2,3, ..
                plane[rank][file] = 1.0 # e.g: (1,2) = b1
            planes.append(plane)

    # ----------------------------------------------------
    # 2. Side-to-move plane (1 plane)
    # ----------------------------------------------------
    stm_plane = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
    planes.append(stm_plane)

    # ----------------------------------------------------
    # 3. Castling rights (4 planes)
    # ----------------------------------------------------
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for right in castling_rights:
        plane = np.ones((8, 8), dtype=np.float32) if right else np.zeros((8, 8), dtype=np.float32)
        planes.append(plane)

    # ----------------------------------------------------
    # 4. En passant plane (1 plane) # chess rule where you can capture pawn that jumped twice
    # ----------------------------------------------------
    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        ep_plane[r][f] = 1.0
    planes.append(ep_plane)

    # final: shape (18, 8, 8)
    return np.stack(planes)

if __name__ == "__main__":
    boards, winner = sample_board(batch_size=1)
    # print(boards)
    print(winner)