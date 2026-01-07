import torch
import chess
import numpy as np
from model import AlphaZeroChessNet
from board import board_to_input_planes
from mcts import AlphaZeroMoveIndexer

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def get_model_move(model, board, move_indexer, device):
    """Get model's move for the current board position"""
    # Convert board to input tensor
    input_planes = board_to_input_planes(board)
    input_tensor = torch.from_numpy(input_planes).unsqueeze(0).to(device)  # (1, 18, 8, 8)
    
    # Run inference
    policy_logits, value = model(input_tensor)
    policy_logits = policy_logits.squeeze(0).cpu().numpy()  # (4672,)
    value = value.item()
    
    # Mask illegal moves
    legal_moves = list(board.legal_moves)
    legal_indices = []
    for move in legal_moves:
        idx = move_indexer.encode(move)
        if idx is not None and 0 <= idx < 4672:
            legal_indices.append((idx, move))
    
    if not legal_indices:
        return None, value
    
    # Get probabilities for legal moves
    legal_probs = []
    for idx, move in legal_indices:
        legal_probs.append((policy_logits[idx], move))
    
    # Take the move with highest probability
    best_move = max(legal_probs, key=lambda x: x[0])[1]
    
    return best_move, value

if __name__ == "__main__":
    import sys
    
    # Load checkpoint
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "checkpoint_7500.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded on {device}")
    
    move_indexer = AlphaZeroMoveIndexer()
    board = chess.Board()
    
    print("\nStarting game. Enter moves in UCI format (e.g., 'e2e4'). Type 'quit' to exit.")
    print("Current board:")
    print(board)
    print()
    
    while True:
        # Get user move
        user_input = input("Your move (UCI): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'reset':
            board = chess.Board()
            print("\nBoard reset:")
            print(board)
            print()
            continue
        
        # Try to parse and push user move
        try:
            user_move = chess.Move.from_uci(user_input)
            if user_move not in board.legal_moves:
                print(f"Invalid move: {user_move}. Legal moves: {[m.uci() for m in board.legal_moves]}")
                continue
            
            board.push(user_move)
            print(f"\nYou played: {user_move.uci()}")
            print(board)
            
            # Check if game is over
            if board.is_game_over():
                print(f"\nGame over: {board.result()}")
                break
            
        except ValueError:
            print(f"Invalid move format: {user_input}. Use UCI format (e.g., 'e2e4')")
            continue
        
        # Get model move
        model_move, value = get_model_move(model, board, move_indexer, device)
        
        if model_move is None:
            print("Model couldn't find a legal move!")
            break
        
        board.push(model_move)
        print(f"\nModel played: {model_move.uci()} (value: {value:.4f})")
        print(board)
        
        # Check if game is over
        if board.is_game_over():
            print(f"\nGame over: {board.result()}")
            break
        
        print()
