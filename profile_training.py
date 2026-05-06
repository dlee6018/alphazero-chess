"""
Profile AlphaZero training pipeline on MPS/CPU.
Generates:
  1. TensorBoard profiler logs (viewable via `tensorboard --logdir=./tb_logs`)
  2. Chrome trace (viewable at chrome://tracing)
  3. Flame graph stacks (viewable at speedscope.app)
  4. Table summary printed to stdout
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import wandb

from model import AlphaZeroChessNet
from board import board_to_input_planes, sample_board
from mcts import AlphaZeroMoveIndexer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHANNELS = 256
N_BLOCKS = 20
BATCH_SIZE = 256
N_MOVES = 4672


def build_training_batch(batch_size=BATCH_SIZE):
    """Simulate building a training batch from MCTS results."""
    move_indexer = AlphaZeroMoveIndexer()

    boards_list, winners = sample_board(batch_size=4)

    board_tensors = []
    policy_targets = []
    value_targets = []

    for game_boards, winner in zip(boards_list, winners):
        for board in game_boards[:batch_size // 4]:
            tensor = board_to_input_planes(board)
            board_tensors.append(tensor)

            # Fake policy target (uniform over legal moves)
            target = np.zeros(N_MOVES, dtype=np.float32)
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                idx = move_indexer.encode(move)
                if idx is not None and 0 <= idx < N_MOVES:
                    target[idx] = 1.0 / len(legal_moves)
            policy_targets.append(target)
            value_targets.append(float(winner))

            if len(board_tensors) >= batch_size:
                break
        if len(board_tensors) >= batch_size:
            break

    # Pad if needed
    while len(board_tensors) < batch_size:
        board_tensors.append(board_tensors[0])
        policy_targets.append(policy_targets[0])
        value_targets.append(value_targets[0])

    board_tensors = board_tensors[:batch_size]
    policy_targets = policy_targets[:batch_size]
    value_targets = value_targets[:batch_size]

    return (
        torch.from_numpy(np.stack(board_tensors)).to(DEVICE),
        torch.from_numpy(np.stack(policy_targets)).to(DEVICE),
        torch.tensor(value_targets, device=DEVICE),
    )


def train_step(model, optimizer, batch_input, policy_targets, value_targets):
    """Single training step matching train.py logic."""
    with record_function("forward_pass"):
        policy_logits, value_logits = model(batch_input)

    with record_function("policy_loss"):
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=1),
            policy_targets,
            reduction='batchmean'
        )

    with record_function("value_loss"):
        soft_targets_map = {
            1.0: torch.tensor([0.05, 0.10, 0.85], device=DEVICE),
            0.0: torch.tensor([0.15, 0.70, 0.15], device=DEVICE),
            -1.0: torch.tensor([0.85, 0.10, 0.05], device=DEVICE),
        }
        value_soft_targets = torch.stack([soft_targets_map[w.item()] for w in value_targets])
        log_probs = F.log_softmax(value_logits.float(), dim=1)
        sample_weights = torch.tensor([3.0 if w.item() != 0.0 else 1.0 for w in value_targets], device=DEVICE)
        value_loss = -(value_soft_targets * log_probs).sum(dim=1)
        value_loss = (value_loss * sample_weights).mean() / sample_weights.mean()

    with record_function("total_loss"):
        loss = policy_loss + value_loss

    with record_function("backward_pass"):
        optimizer.zero_grad()
        loss.backward()

    with record_function("optimizer_step"):
        optimizer.step()

    return policy_loss.item(), value_loss.item(), loss.item()


def inference_step(model, batch_input):
    """Single inference step matching inference_worker logic."""
    with torch.no_grad():
        with record_function("inference_forward"):
            policy_logits, value_logits = model(batch_input)
        with record_function("value_conversion"):
            value = model.value_head.logits_to_expected_value(value_logits)
        with record_function("cpu_transfer"):
            policy_cpu = policy_logits.cpu()
            value_cpu = value.cpu()
    return policy_cpu, value_cpu


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {CHANNELS} channels, {N_BLOCKS} residual blocks")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    model = AlphaZeroChessNet(channels=CHANNELS, n_blocks=N_BLOCKS, n_moves=N_MOVES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Parameters: {model.count_parameters():,}")
    print()

    # TensorBoard scalar/histogram/graph writer (separate from train.py's logs)
    writer = SummaryWriter("./tb_logs/profile")
    writer.add_graph(model, torch.randn(1, 18, 8, 8).to(DEVICE))
    print("✓ Model graph written to TensorBoard")

    # wandb init for profiling run
    wandb.init(
        project="alphazero-chess",
        job_type="profile",
        config={
            "channels": CHANNELS,
            "n_blocks": N_BLOCKS,
            "batch_size": BATCH_SIZE,
            "n_moves": N_MOVES,
            "device": str(DEVICE),
        },
    )
    print("✓ wandb run initialized")

    # Build batch
    print("Building training batch from dataset...")
    batch_input, policy_targets, value_targets = build_training_batch()
    print(f"Batch shape: {batch_input.shape}")
    print()

    # Warmup
    print("Warming up (5 steps)...")
    model.train()
    for _ in range(5):
        train_step(model, optimizer, batch_input, policy_targets, value_targets)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    print()

    # === PROFILE TRAINING ===
    print("=" * 60)
    print("PROFILING TRAINING (schedule: 1 wait, 2 warmup, 5 active, repeat=1)")
    print("=" * 60)

    activities = [ProfilerActivity.CPU]

    tb_handler = tensorboard_trace_handler("./tb_logs/training")

    N_PROFILED_STEPS = 8   # wait(1) + warmup(2) + active(5) = 8
    N_EXTRA_STEPS = 50     # additional steps for scalar/histogram data
    N_TOTAL_STEPS = N_PROFILED_STEPS + N_EXTRA_STEPS

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=tb_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(N_PROFILED_STEPS):
            with record_function(f"train_step_{step}"):
                p_loss, v_loss, total_loss = train_step(model, optimizer, batch_input, policy_targets, value_targets)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            prof.step()

            writer.add_scalar("loss/policy", p_loss, step)
            writer.add_scalar("loss/value", v_loss, step)
            writer.add_scalar("loss/total", total_loss, step)
            wandb.log({"loss/policy": p_loss, "loss/value": v_loss, "loss/total": total_loss}, step=step)

    # Continue training beyond profiler for more scalar/histogram data
    print(f"Profiling done. Running {N_EXTRA_STEPS} more steps for scalar/histogram data...")
    for step in range(N_PROFILED_STEPS, N_TOTAL_STEPS):
        p_loss, v_loss, total_loss = train_step(model, optimizer, batch_input, policy_targets, value_targets)

        writer.add_scalar("loss/policy", p_loss, step)
        writer.add_scalar("loss/value", v_loss, step)
        writer.add_scalar("loss/total", total_loss, step)
        wandb.log({"loss/policy": p_loss, "loss/value": v_loss, "loss/total": total_loss}, step=step)

        if step % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, step)

    # Print table
    print("\n--- Training: Top 20 operations by CPU time ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("\n--- Training: Top 20 operations by Self CPU time ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Export stacks for flame graph (chrome trace already saved by tb_handler)
    prof.export_stacks("profile_training_stacks.txt", metric="self_cpu_time_total")
    print("\n✓ TensorBoard logs saved to: ./tb_logs/training/")
    print("  Run: tensorboard --logdir=./tb_logs")
    print("✓ Flame graph stacks saved to: profile_training_stacks.txt")

    # === PROFILE INFERENCE ===
    print()
    print("=" * 60)
    print("PROFILING INFERENCE (schedule: 1 wait, 2 warmup, 5 active, repeat=1)")
    print("=" * 60)

    model.eval()
    # Warmup inference
    for _ in range(3):
        inference_step(model, batch_input)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    tb_handler_inf = tensorboard_trace_handler("./tb_logs/inference")

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=tb_handler_inf,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof_inf:
        for step in range(8):
            with record_function(f"inference_step_{step}"):
                inference_step(model, batch_input)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            prof_inf.step()

    print("\n--- Inference: Top 20 operations by CPU time ---")
    print(prof_inf.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    prof_inf.export_stacks("profile_inference_stacks.txt", metric="self_cpu_time_total")
    print("\n✓ TensorBoard logs saved to: ./tb_logs/inference/")
    print("✓ Flame graph stacks saved to: profile_inference_stacks.txt")

    # === THROUGHPUT BENCHMARK ===
    print()
    print("=" * 60)
    print("THROUGHPUT BENCHMARK")
    print("=" * 60)

    # Training throughput
    model.train()
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    n_steps = 20
    for _ in range(n_steps):
        _ = train_step(model, optimizer, batch_input, policy_targets, value_targets)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    train_time = t1 - t0
    print(f"Training: {n_steps} steps in {train_time:.2f}s")
    print(f"  {n_steps * BATCH_SIZE / train_time:.0f} samples/sec")
    print(f"  {train_time / n_steps * 1000:.1f} ms/step")

    # Inference throughput
    model.eval()
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    n_inf = 50
    for _ in range(n_inf):
        inference_step(model, batch_input)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    inf_time = t1 - t0
    print(f"\nInference: {n_inf} steps in {inf_time:.2f}s")
    print(f"  {n_inf * BATCH_SIZE / inf_time:.0f} samples/sec")
    print(f"  {inf_time / n_inf * 1000:.1f} ms/step")

    # === MCTS BOTTLENECK ESTIMATE ===
    print()
    print("=" * 60)
    print("MCTS PIPELINE ESTIMATE")
    print("=" * 60)
    single_inf_ms = inf_time / n_inf * 1000
    sims_per_move = 200
    batches_per_move = sims_per_move / 64  # batch_size=64 in MCTS
    time_per_move = batches_per_move * single_inf_ms
    print(f"  Single inference batch ({BATCH_SIZE} samples): {single_inf_ms:.1f} ms")
    print(f"  MCTS sims/move: {sims_per_move}, batch_size: 64")
    print(f"  Estimated neural evals per move: {batches_per_move:.0f}")
    print(f"  Estimated time per move (inference only): {time_per_move:.0f} ms")
    print("  With 8 workers sharing 1 inference process: throughput limited by GPU batching")

    # Log throughput as wandb summary metrics
    wandb.summary["benchmark/train_samples_per_sec"] = n_steps * BATCH_SIZE / train_time
    wandb.summary["benchmark/train_ms_per_step"] = train_time / n_steps * 1000
    wandb.summary["benchmark/inference_samples_per_sec"] = n_inf * BATCH_SIZE / inf_time
    wandb.summary["benchmark/inference_ms_per_step"] = inf_time / n_inf * 1000

    writer.close()
    wandb.finish()
    print("\n✓ All TensorBoard + wandb data written. Run: tensorboard --logdir=./tb_logs")


if __name__ == "__main__":
    main()
