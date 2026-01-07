from board import board_to_input_planes, sample_board
from mcts import MCTS, AlphaZeroMoveIndexer
from model import AlphaZeroChessNet
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import time
import chess
import os

def mcts_worker_supervised(worker_id, inference_queue, response_queues, result_queue, c_puct, n_simulations):
    """Worker process that continuously generates MCTS data"""

    response_queue = response_queues[worker_id]
    mcts = MCTS(c_puct=c_puct, n_simulations=n_simulations, batch_size=64,
                inference_queue=inference_queue, inference_result_queue=response_queue, worker_id=worker_id)
    
    while True: 
        try:
            boards_list, winners = sample_board(batch_size=1)
            boards = boards_list[0]
            winner = winners[0]
            
            for board in boards:
                move, probs = mcts.search(board)
                result_queue.put((board, move, probs, winner), timeout=1.0)
            
        except queue.Full:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(0.1)
            
def sample_move_with_temperature(probs: dict, temperature: float):
    """Sample a move from MCTS probability distribution with temperature scaling.

    Args:
        probs: dict mapping moves to visit count probabilities
        temperature: temperature for sampling (1.0 = proportional to probs,
                     0 = deterministic/argmax, >1 = more uniform)
    Returns:
        Selected move
    """
    moves = list(probs.keys())
    visit_probs = np.array([probs[m] for m in moves], dtype=np.float64)

    if temperature == 0:
        # Deterministic: pick highest probability move
        return moves[np.argmax(visit_probs)]

    # Apply temperature: p_i^(1/T) / sum(p_j^(1/T))
    scaled = visit_probs ** (1.0 / temperature)
    scaled /= scaled.sum()

    return moves[np.random.choice(len(moves), p=scaled)]


def mcts_worker_self_play(worker_id, inference_queue, response_queues, result_queue, c_puct, n_simulations):
    """Worker process that generates board positions via self play"""

    response_queue = response_queues[worker_id]
    mcts = MCTS(c_puct=c_puct, n_simulations=n_simulations, batch_size=64,
                inference_queue=inference_queue, inference_result_queue=response_queue, worker_id=worker_id)

    while True:
        try:
            board = chess.Board()
            game_history = []
            move_count = 0

            while not board.is_game_over():
                _, probs = mcts.search(board)

                # Temperature scaling: high temp early for exploration, low temp later for strength
                # First 30 plies (~15 moves): temperature=1.0 for diverse openings
                # After that: temperature=0.1 for near-deterministic play
                if move_count < 30:
                    temperature = 1.0
                else:
                    temperature = 0.1

                move = sample_move_with_temperature(probs, temperature)
                game_history.append((board.copy(), move, probs))
                board.push(move)
                move_count += 1
            
            result = board.result()
            if result == "1-0":
                final_winner = 1.0  # White won
            elif result == "0-1":
                final_winner = -1.0  # Black won
            else:
                final_winner = 0.0  # Draw

            game_data = [
                (board_pos, move, probs, 
                 final_winner if board_pos.turn == chess.WHITE else -final_winner)
                for board_pos, move, probs in game_history
            ]
            while game_data:
                remaining = []
                for item in game_data:
                    try:
                        result_queue.put_nowait(item)
                    except queue.Full:
                        remaining.append(item)
                
                if remaining:
                    game_data = remaining
                    time.sleep(0.01)  
                else:
                    break
            
        except queue.Full:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(0.1)

@torch.no_grad()
def inference_worker(inference_queue, response_queues, weight_update_queue, model_state_dict, device_str, max_batch_size=512, max_wait_ms=20):
    """Single GPU worker that batches and processes inference requests for maximum GPU utilization"""
    device = torch.device(device_str)
    model = AlphaZeroChessNet(channels=128, n_blocks=20, n_moves=4672).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    while True:
        try:
            try:
                new_state_dict = weight_update_queue.get_nowait()
                model.load_state_dict(new_state_dict)
            except queue.Empty:
                pass

            pending_requests = []  # List of (worker_id, batch_tensor, request_id, batch_size)
            total_samples = 0

            # Get first request (blocking)
            try:
                worker_id, batch_tensor, request_id = inference_queue.get(timeout=0.1)
                batch_size = batch_tensor.shape[0]
                pending_requests.append((worker_id, batch_tensor, request_id, batch_size))
                total_samples += batch_size
            except queue.Empty:
                continue

            # Collect more requests (non-blocking) to fill up the batch
            deadline = time.time() + max_wait_ms / 1000.0
            while total_samples < max_batch_size and time.time() < deadline:
                try:
                    worker_id, batch_tensor, request_id = inference_queue.get_nowait()
                    batch_size = batch_tensor.shape[0]
                    pending_requests.append((worker_id, batch_tensor, request_id, batch_size))
                    total_samples += batch_size
                except queue.Empty:
                    break

            if not pending_requests:
                continue

            all_tensors = [req[1] for req in pending_requests]
            mega_batch = np.concatenate(all_tensors, axis=0)
            mega_batch_tensor = torch.from_numpy(mega_batch).to(device)

            policy_logits, value_logits = model(mega_batch_tensor)

            # Convert 3-class value logits to expected value for MCTS
            value = model.value_head.logits_to_expected_value(value_logits)

            # Convert to CPU numpy for IPC
            policy_logits_cpu = policy_logits.cpu()
            value_cpu = value.cpu()

            offset = 0
            # extract the correct batch and send back to the worker
            for worker_id, _, request_id, batch_size in pending_requests:
                policy_slice = policy_logits_cpu[offset:offset + batch_size]
                value_slice = value_cpu[offset:offset + batch_size]
                offset += batch_size

                response_queues[worker_id].put((request_id, policy_slice, value_slice))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Inference worker error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

CHECKPOINT_STEPS = 500

def train_batch(batch_buffer, model, optimizer, device, move_indexer, step):
    """Train on a batch of MCTS results"""
    if len(batch_buffer) == 0:
        return None

    boards, _, probs_list, winners = zip(*batch_buffer)

    board_tensors = [board_to_input_planes(board) for board in boards]
    curr_batch = np.stack(board_tensors)
    curr_batch = torch.from_numpy(curr_batch).to(device)

    with autocast(device_type=device.type):
        policy_logits, value_logits = model(curr_batch)  # (batch_size, 4672), (batch_size, 3)

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

        policy_targets = torch.stack(policy_targets)  # (batch_size, 4672)

        # Policy loss: KL divergence (better for probability distributions)
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=1),
            policy_targets,
            reduction='batchmean'
        )

        # Value loss: Cross-entropy with 3 classes (loss, draw, win)
        # Convert winner values (-1, 0, 1) to class indices (0, 1, 2)
        winners_tensor = torch.tensor(winners, dtype=torch.float32, device=device)
        value_class_targets = (winners_tensor + 1).long()  # -1->0, 0->1, 1->2

        # Class weights: weight decisive games (win/loss) higher than draws
        # This prevents the model from always predicting "draw"
        class_weights = torch.tensor([3.0, 1.0, 3.0], device=device)  # [loss, draw, win]
        value_loss = F.cross_entropy(value_logits.float(), value_class_targets, weight=class_weights)

        # Combined loss
        loss = policy_loss + value_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Compute value prediction stats for monitoring
    with torch.no_grad():
        value_probs = F.softmax(value_logits.float(), dim=1)
        pred_loss_pct = (value_probs[:, 0].mean() * 100).item()
        pred_draw_pct = (value_probs[:, 1].mean() * 100).item()
        pred_win_pct = (value_probs[:, 2].mean() * 100).item()

    print(f"Step {step}: Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Total loss: {loss.item():.4f} | Value preds: L={pred_loss_pct:.1f}% D={pred_draw_pct:.1f}% W={pred_win_pct:.1f}%", flush=True)

    if step % CHECKPOINT_STEPS == 0:
        checkpoint_path = f"checkpoint_{step}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    return model_state_dict


    

def simple_train_supervised(batch_size: int = 256, num_workers: int = 4, checkpoint_path: str = "checkpoint_8000.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroChessNet(channels=128, n_blocks=20, n_moves=4672).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Successfully loaded {checkpoint_path}", flush=True)

    move_indexer = AlphaZeroMoveIndexer()

    # Create shared queues
    result_queue = Queue(maxsize=10000)
    weight_update_queue = Queue(maxsize=num_workers * 2)
    inference_weight_update_queue = Queue(maxsize=2)  
    inference_queue = Queue(maxsize=10000)  

    # Create per-worker response queues
    response_queues = [Queue(maxsize=100) for _ in range(num_workers)]

    # Start single inference worker (GPU)
    # Convert to CPU before sending through multiprocessing
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    inference_worker_process = Process(target=inference_worker,
                                       args=(inference_queue, response_queues, inference_weight_update_queue, model_state_dict, str(device)))
    inference_worker_process.start()

    # Start MCTS worker processes (CPU)
    workers = []
    for worker_id in range(num_workers):
        p = Process(target=mcts_worker_self_play,
                   args=(worker_id, inference_queue, response_queues, result_queue,
                         1.5, 200))
        p.start()
        workers.append(p)
    
    batch_buffer = []
    step = 0

    try:
        while True:
            try:

                board, move, probs, winner = result_queue.get(timeout=0.1)
                batch_buffer.append((board, move, probs, winner))

                if len(batch_buffer) >= batch_size:
                    step += 1
                    updated_state_dict = train_batch(batch_buffer, model, optimizer, device, move_indexer, step)
                    if updated_state_dict is not None:
                        try:
                            inference_weight_update_queue.put_nowait(updated_state_dict)
                        except queue.Full:
                            pass
                        for _ in range(num_workers):
                            try:
                                weight_update_queue.put_nowait(updated_state_dict)
                            except queue.Full:
                                pass  # Skip if queue full (workers will get next update)
                    batch_buffer = []
                    
            except queue.Empty:
                if len(batch_buffer) > 0:
                    step += 1
                    updated_state_dict = train_batch(batch_buffer, model, optimizer, device, move_indexer, step)
                    if updated_state_dict is not None:
                        try:
                            inference_weight_update_queue.put_nowait(updated_state_dict)
                        except queue.Full:
                            pass
                        for _ in range(num_workers):
                            try:
                                weight_update_queue.put_nowait(updated_state_dict)
                            except queue.Full:
                                pass  # Skip if queue full (workers will get next update)
                    batch_buffer = []
                    
    except KeyboardInterrupt:
        print("Shutting down...")
        inference_worker_process.terminate()
        inference_worker_process.join()
        for p in workers:
            p.terminate()
            p.join()
        print("Workers terminated")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    simple_train_supervised(num_workers=20) # 20 workers for self play
