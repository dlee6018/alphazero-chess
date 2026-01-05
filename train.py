from board import board_to_input_planes, sample_board
from mcts import MCTS, AlphaZeroMoveIndexer
from model import AlphaZeroChessNet
import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import time


def mcts_worker(result_queue, weight_update_queue, model_state_dict, device_str, c_puct, n_simulations, batch_size):
    """Worker process that continuously generates MCTS data"""
    device = torch.device(device_str)
    model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    mcts = MCTS(c_puct=c_puct, n_simulations=n_simulations, batch_size=batch_size, model=model)
    
    while True:  # Continuously work
        try:
            # Check for weight updates (non-blocking)
            try:
                new_state_dict = weight_update_queue.get_nowait()
                model.load_state_dict(new_state_dict)
                mcts.model = model  # Update MCTS model reference
            except queue.Empty:
                pass
            
            # Sample board
            boards, winner = sample_board(batch_size=1)
            
            # Process all boards from the game
            for board in boards:
                # Run MCTS
                move, probs = mcts.search(board)
                
                # Put in queue (blocks if queue full - backpressure)
                result_queue.put((board, move, probs, winner), timeout=1.0)
            
        except queue.Full:
            # Queue full - wait a bit and retry
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Log error and continue
            print(f"Worker error: {e}")
            time.sleep(0.1)


def train_batch(batch_buffer, model, optimizer, device, move_indexer):
    """Train on a batch of MCTS results"""
    if len(batch_buffer) == 0:
        return None
    
    # Extract data
    boards, moves, probs_list, winners = zip(*batch_buffer)
    
    # Convert boards to input tensors
    board_tensors = [board_to_input_planes(board) for board in boards]
    curr_batch = np.stack(board_tensors)
    curr_batch = torch.from_numpy(curr_batch).to(device)
    
    # Get model predictions
    with autocast(device_type=device.type):
        policy_logits, value = model(curr_batch)  # (batch_size, 4672), (batch_size, 1)
        
        # Convert MCTS probability distributions to target tensors
        policy_targets = []
        for prob_dict in probs_list:
            # Create target distribution: (4672,) tensor with probabilities
            target = torch.zeros(4672, dtype=torch.float32, device=device)
            for move, prob in prob_dict.items():
                move_idx = move_indexer.encode(move)
                if move_idx is not None and 0 <= move_idx < 4672:
                    target[move_idx] = prob
            policy_targets.append(target)
        
        
        print("policy_logits", policy_logits)
        print("policy_targets", policy_targets)
        policy_targets = torch.stack(policy_targets)  # (batch_size, 4672)
        
        # Policy loss: KL divergence (better for probability distributions)
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=1),
            policy_targets,
            reduction='batchmean'
        )
        
        # Value loss: use actual game outcome
        value_targets = torch.tensor(winners, dtype=torch.float32, device=device).unsqueeze(1)  # (batch_size, 1)
        value_loss = F.mse_loss(value, value_targets)
        
        # Combined loss
        loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update model weights for workers
    model_state_dict = model.state_dict()
    
    print(f"Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Total loss: {loss.item():.4f}")
    
    return model_state_dict


def simple_train_supervised(batch_size: int = 256, num_workers: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    move_indexer = AlphaZeroMoveIndexer()
    
    # Create shared queues
    result_queue = Queue(maxsize=1000)
    weight_update_queue = Queue(maxsize=num_workers * 2)
    
    # Start worker processes
    model_state_dict = model.state_dict()
    workers = []
    for _ in range(num_workers):
        p = Process(target=mcts_worker, 
                   args=(result_queue, weight_update_queue, model_state_dict, str(device), 
                         1.5, 200, 256))
        p.start()
        workers.append(p)
    
    # Main training loop
    batch_buffer = []
    
    try:
        while True:
            try:
                # Get result (non-blocking)
                board, move, probs, winner = result_queue.get(timeout=0.1)
                batch_buffer.append((board, move, probs, winner))
                
                # Train when batch ready
                if len(batch_buffer) >= batch_size:
                    updated_state_dict = train_batch(batch_buffer, model, optimizer, device, move_indexer)
                    if updated_state_dict is not None:
                        # Send updated weights to all workers
                        for _ in range(num_workers):
                            try:
                                weight_update_queue.put_nowait(updated_state_dict)
                            except queue.Full:
                                pass  # Skip if queue full (workers will get next update)
                    batch_buffer = []
                    
            except queue.Empty:
                # Queue empty - train partial batch if large enough
                if len(batch_buffer) >= batch_size // 2:
                    updated_state_dict = train_batch(batch_buffer, model, optimizer, device, move_indexer)
                    if updated_state_dict is not None:
                        # Send updated weights to all workers
                        for _ in range(num_workers):
                            try:
                                weight_update_queue.put_nowait(updated_state_dict)
                            except queue.Full:
                                pass  # Skip if queue full (workers will get next update)
                    batch_buffer = []
                    
    except KeyboardInterrupt:
        # Graceful shutdown
        print("Shutting down...")
        for p in workers:
            p.terminate()
            p.join()
        print("Workers terminated")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    simple_train_supervised()
