from board import board_to_input_planes, sample_board
from mcts import MCTS
import numpy as np
import torch
import torch.nn.functional as F

def simple_train_supervised(batch_size: int = 4):
    batches = []
    board_objects = []
    winners = []
    mcts = MCTS(c_puct=1.5, n_simulations=200)
    move_indexer = mcts.move_indexer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mcts.model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    while True:  # Continue training loop
        # If batches is empty, sample a new board
        if len(batches) == 0:
            boards, winner = sample_board(batch_size=1)
            for board in boards:
                batch = board_to_input_planes(board)
                batches.append(batch)
                board_objects.append(board)
                winners.append(winner)
        
        # Take up to batch_size items from batches
        k = min(batch_size, len(batches))
        curr_batch = batches[:k]
        curr_boards = board_objects[:k]
        curr_winners = winners[:k]
        batches = batches[k:]  # Remove used batches
        board_objects = board_objects[k:]  # Remove used boards
        winners = winners[k:]  # Remove used winners
        
        curr_batch = np.stack(curr_batch)
        curr_batch = torch.from_numpy(curr_batch)
        
        # Get model predictions
        policy_logits, value = model(curr_batch)  # (batch_size, 4672), (batch_size, 1)
        
        # Get MCTS search results
        moves, probs = [], []
        for i, board in enumerate(curr_boards):
            move, prob = mcts.search(board)
            moves.append(move)
            probs.append(prob)
        
        # Convert MCTS probability distributions to target tensors
        policy_targets = []
        for prob_dict in probs:
            # Create target distribution: (4672,) tensor with probabilities
            target = torch.zeros(4672, dtype=torch.float32)
            for move, prob in prob_dict.items():
                move_idx = move_indexer.encode(move)
                if move_idx is not None and 0 <= move_idx < 4672:
                    target[move_idx] = prob
            policy_targets.append(target)
        
        policy_targets = torch.stack(policy_targets)  # (batch_size, 4672)
        
        # Policy loss: KL divergence (better for probability distributions)
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=1),
            policy_targets,
            reduction='batchmean'
        )
        
        # Value loss: use actual game outcome
        value_targets = torch.tensor(curr_winners, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)
        value_loss = F.mse_loss(value, value_targets)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Total loss: {loss.item():.4f}")
     
    
simple_train_supervised()