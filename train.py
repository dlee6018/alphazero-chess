from board import board_to_input_planes, sample_board
from mcts import MCTS, AlphaZeroMoveIndexer
from model import AlphaZeroChessNet
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import time
import chess
import os
import wandb
import traceback
import math
from stockfish import Stockfish

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
STOCKFISH_DEPTH = 12

def cp_to_value(cp: int) -> float:
    """Convert centipawn score to [-1, 1] using tanh scaling."""
    return math.tanh(cp / 400.0)

def supervised_data_worker(worker_id, result_queue):
    """Worker that generates training data with Stockfish evaluations."""
    sf = Stockfish(path=STOCKFISH_PATH, depth=STOCKFISH_DEPTH)
    sf.update_engine_parameters({"Threads": 1, "Hash": 64})

    while True:
        try:
            boards_list, winners = sample_board(batch_size=1)
            boards = boards_list[0]

            for i in range(len(boards) - 1):
                board = boards[i]
                if board.is_game_over():
                    continue

                sf.set_fen_position(board.fen())
                eval_result = sf.get_evaluation()
                if eval_result["type"] == "cp":
                    cp = eval_result["value"]
                elif eval_result["type"] == "mate":
                    cp = 10000 if eval_result["value"] > 0 else -10000
                else:
                    continue

                # Policy target: Stockfish's best move
                best_move_uci = sf.get_best_move()
                if best_move_uci is None:
                    continue
                best_move = chess.Move.from_uci(best_move_uci)
                probs = {best_move: 1.0}

                # Value target: Stockfish eval normalized to [-1, 1]
                value = cp_to_value(cp)

                try:
                    result_queue.put((board, best_move, probs, value), timeout=1.0)
                except queue.Full:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Supervised worker {worker_id} error: {e}")
            time.sleep(0.1)


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
    total = scaled.sum()
    if total == 0:
        scaled = np.ones_like(scaled) / len(scaled)
    else:
        scaled /= total

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
                move, probs = mcts.search(board)

                if not probs:
                    if move is None:
                        break
                    game_history.append((board.copy(), move, {}))
                    board.push(move)
                    move_count += 1
                    continue

                if move_count < 30:
                    temperature = 1.5
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
    model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
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

            policy_logits, value = model(mega_batch_tensor)

            # Convert to CPU for IPC (value is already scalar in [-1, 1])
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

# ─── Config toggles ───────────────────────────────────────────────────────────
MODE = "supervised"  # "supervised" (fast, from game DB) or "self_play" (slow, MCTS self-play)
VALUE_LOSS_WEIGHT = 1.0
CHECKPOINT_STEPS = 500
MAX_STEPS = 1000

def train_batch(batch_buffer, model, optimizer, device, move_indexer, step, writer=None):
    """Train on a batch of MCTS results"""
    assert len(batch_buffer) == 256, f"Expected batch size 256, got {len(batch_buffer)}"

    boards, _, probs_list, winners = zip(*batch_buffer)

    board_tensors = [board_to_input_planes(board) for board in boards]
    curr_batch = np.stack(board_tensors)
    curr_batch = torch.from_numpy(curr_batch).to(device)

    with autocast(device_type=device.type):
        policy_logits, value_pred = model(curr_batch)  # (batch_size, 4672), (batch_size, 1)

        # Convert MCTS probability distributions to target tensors
        policy_targets = []
        for prob_dict in probs_list:
            target = torch.zeros(4672, dtype=torch.float32, device=device)
            for move, prob in prob_dict.items():
                move_idx = move_indexer.encode(move)
                if move_idx is not None and 0 <= move_idx < 4672:
                    target[move_idx] = prob
            policy_targets.append(target)

        policy_targets = torch.stack(policy_targets)  # (batch_size, 4672)

        # Policy loss: KL divergence
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=1),
            policy_targets,
            reduction='batchmean'
        )

        # Value loss: MSE against Stockfish evaluation (scalar in [-1, 1])
        value_targets = torch.tensor(list(winners), dtype=torch.float32, device=device).unsqueeze(1)
        value_loss = F.mse_loss(value_pred, value_targets)

        # Combined loss
        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
    
    optimizer.zero_grad()
    loss.backward()

    # Compute gradient norms before optimizer step (gradients exist now)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    policy_params = [p for p in model.policy_head.parameters() if p.grad is not None]
    policy_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in policy_params]))
    value_params = [p for p in model.value_head.parameters() if p.grad is not None]
    value_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in value_params]))
    total_norm = torch.norm(torch.stack([policy_norm, value_norm]))

    # Per-residual-block gradient norms (detect backbone gradient starvation)
    block_grad_norms = {}
    for name, param in model.named_parameters():
        if "res_blocks" in name and param.grad is not None:
            block_idx = name.split(".")[1]
            key = f"block_{block_idx}"
            if key not in block_grad_norms:
                block_grad_norms[key] = []
            block_grad_norms[key].append(torch.norm(param.grad).item())
    block_grad_norms_agg = {k: np.mean(v) for k, v in block_grad_norms.items()}

    optimizer.step()

    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # ─── Compute metrics ───────────────────────────────────────────────────────
    with torch.no_grad():
        pred_mean = value_pred.mean().item()
        pred_std = value_pred.std().item()
        target_mean = value_targets.mean().item()
        target_std = value_targets.std().item()
        correlation = torch.corrcoef(torch.stack([value_pred.squeeze(), value_targets.squeeze()]))[0, 1].item()

    print(f"Step {step}: P_loss={policy_loss.item():.4f} V_loss={value_loss.item():.4f} Total={loss.item():.4f} | Pred: mean={pred_mean:.3f} std={pred_std:.3f} | Target: mean={target_mean:.3f} std={target_std:.3f} | Corr={correlation:.3f}", flush=True)

    # ─── Logging (all metrics grouped by category) ────────────────────────────
    log_dict = {
        # Loss metrics
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/total": loss.item(),

        # Value head predictions (scalar)
        "value/pred_mean": pred_mean,
        "value/pred_std": pred_std,
        "value/target_mean": target_mean,
        "value/target_std": target_std,
        "value/correlation": correlation,

        # Gradient health
        "gradients/total_norm": total_norm.item(),
        "gradients/policy_head_norm": policy_norm.item(),
        "gradients/value_head_norm": value_norm.item(),

        # Training config
        "lr": optimizer.param_groups[0]['lr'],
        "batch/size": len(batch_buffer),
    }

    # Per-block backbone gradient norms
    for block_name, norm_val in block_grad_norms_agg.items():
        log_dict[f"gradients/backbone_{block_name}"] = norm_val

    # MPS memory (every 50 steps)
    if step % 50 == 0 and device.type == "mps":
        log_dict["mps/allocated_mb"] = torch.mps.current_allocated_memory() / 1024**2
        log_dict["mps/driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024**2

    wandb.log(log_dict, step=step)

    # TensorBoard (histograms every 50 steps)
    if writer is not None:
        writer.add_scalar("loss/policy", policy_loss.item(), step)
        writer.add_scalar("loss/value", value_loss.item(), step)
        writer.add_scalar("loss/total", loss.item(), step)
        if step % 50 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, step)

    # ─── Checkpointing ────────────────────────────────────────────────────────
    if step % CHECKPOINT_STEPS == 0:
        checkpoint_path = f"checkpoint_{step}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        artifact = wandb.Artifact(f"model-checkpoint-{step}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Saved checkpoint to {checkpoint_path} (+ wandb artifact)", flush=True)

    return model_state_dict


def simple_train_supervised(batch_size: int = 256, num_workers: int = 4, checkpoint_path: str = None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # wandb init
    wandb.init(
        project="alphazero-chess",
        config={
            "mode": MODE,
            "channels": 256,
            "n_blocks": 20,
            "n_moves": 4672,
            "batch_size": batch_size,
            "lr": 0.001,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "optimizer": "adam",
            "num_workers": num_workers,
            "n_simulations": 50 if MODE == "self_play" else 0,
            "c_puct": 1.5,
            "checkpoint_path": checkpoint_path,
            "device": str(device),
        },
    )

    writer = SummaryWriter("./tb_logs/train")

    model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-5)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Successfully loaded {checkpoint_path}", flush=True)

    wandb.watch(model, log="all", log_freq=500)

    # Log model graph for the "Graphs" tab
    dummy_input = torch.randn(1, 18, 8, 8).to(device)
    writer.add_graph(model, dummy_input)

    move_indexer = AlphaZeroMoveIndexer()

    # Create shared queues
    result_queue = Queue(maxsize=10000)
    inference_worker_process = None
    workers = []

    if MODE == "self_play":
        weight_update_queue = Queue(maxsize=num_workers * 2)
        inference_weight_update_queue = Queue(maxsize=2)
        inference_queue = Queue(maxsize=10000)
        response_queues = [Queue(maxsize=100) for _ in range(num_workers)]

        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        inference_worker_process = Process(target=inference_worker,
                                           args=(inference_queue, response_queues, inference_weight_update_queue, model_state_dict, str(device)))
        inference_worker_process.start()

        for worker_id in range(num_workers):
            p = Process(target=mcts_worker_self_play,
                       args=(worker_id, inference_queue, response_queues, result_queue,
                             1.5, 50))
            p.start()
            workers.append(p)
    else:
        for worker_id in range(num_workers):
            p = Process(target=supervised_data_worker,
                       args=(worker_id, result_queue))
            p.start()
            workers.append(p)

    import random
    shuffle_buffer = []
    shuffle_buffer_size = batch_size * 10  # 2560 positions (~42 games worth)
    step = 0
    last_step_time = time.time()
    last_progress_time = time.time()

    print(f"[{MODE}] Filling shuffle buffer ({shuffle_buffer_size} positions) before training...", flush=True)

    exit_code = 0
    try:
        while True:
            try:
                board, move, probs, winner = result_queue.get(timeout=0.1)
                shuffle_buffer.append((board, move, probs, winner))

                if step == 0 and time.time() - last_progress_time > 10:
                    print(f"  Shuffle buffer: {len(shuffle_buffer)}/{shuffle_buffer_size}", flush=True)
                    last_progress_time = time.time()

                # Once buffer is full, shuffle and pop a batch
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    step += 1
                    if step > MAX_STEPS:
                        print(f"Reached max steps ({MAX_STEPS}), stopping.", flush=True)
                        raise KeyboardInterrupt

                    random.shuffle(shuffle_buffer)
                    batch_buffer = shuffle_buffer[:batch_size]
                    shuffle_buffer = shuffle_buffer[batch_size:]

                    now = time.time()
                    step_duration = now - last_step_time
                    last_step_time = now
                    if step > 1:
                        samples_per_sec = batch_size / step_duration
                        if writer is not None:
                            writer.add_scalar("throughput/step_duration_sec", step_duration, step)
                            writer.add_scalar("throughput/samples_per_sec", samples_per_sec, step)
                        wandb.log({
                            "throughput/step_duration_sec": step_duration,
                            "throughput/samples_per_sec": samples_per_sec,
                        }, step=step)
                    updated_state_dict = train_batch(batch_buffer, model, optimizer, device, move_indexer, step, writer)
                    scheduler.step()
                    if updated_state_dict is not None and MODE == "self_play":
                        try:
                            inference_weight_update_queue.put_nowait(updated_state_dict)
                        except queue.Full:
                            pass

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        exit_code = 1
        print(f"Training crashed with error: {e}", flush=True)
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        wandb.finish(exit_code=exit_code)
        if inference_worker_process:
            inference_worker_process.terminate()
            inference_worker_process.join()
        for p in workers:
            p.terminate()
            p.join()
        print("Workers terminated")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    simple_train_supervised(num_workers=8) # 20 workers for self play
