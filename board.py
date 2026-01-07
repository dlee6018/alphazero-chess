import chess
import numpy as np
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "games.csv" # gets file path to game.csv in curr working directory

_df = None

def sample_board(batch_size: int = 1):
    """
    Samples board(s) from the chess dataset and returns the board states along with the winner.
    Returns:
        tuple: (boards_list, winners) where:
            - boards_list: list of lists of chess.Board objects (one list per game)
            - winners: list of int (1 if white won, -1 if black won, 0 if tied)
    """
    global _df
    if _df is None:
        _df = pd.read_csv(file_path)
    ex = _df.sample(batch_size)
    
    boards_list = []
    winners = []
    
    for idx in range(len(ex)):
        row = ex.iloc[idx]
        
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
        
        boards_list.append(boards)
        winners.append(winner)
    
    return boards_list, winners

def board_to_input_planes(board: chess.Board):
    """
    Returns a tensor of shape (18, 8, 8)
    """
    planes = []

    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]

    for color in [chess.WHITE, chess.BLACK]:
        for piece in piece_order:
            plane = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece, color): # set of squares indices from 0-63
                rank = 7 - chess.square_rank(square)   # a,b,c, ..
                file = chess.square_file(square) # 1,2,3, ..
                plane[rank][file] = 1.0 
            planes.append(plane)

    stm_plane = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
    planes.append(stm_plane)

    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for right in castling_rights:
        plane = np.ones((8, 8), dtype=np.float32) if right else np.zeros((8, 8), dtype=np.float32)
        planes.append(plane)

    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        ep_plane[r][f] = 1.0
    planes.append(ep_plane)

    # final: shape (18, 8, 8)
    return np.stack(planes)

if __name__ == "__main__":
    boards_list, winners = sample_board(batch_size=2)
    print(f"Number of games: {len(boards_list)}")
    for i, (boards, winner) in enumerate(zip(boards_list, winners)):
        print(f"Game {i+1}: {len(boards)} board states, winner: {winner}")