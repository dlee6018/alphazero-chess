"""
Export wandb run data to CSV for offline analysis.

Usage:
  .venv/bin/python export_wandb.py <run_path>

  run_path format: "entity/project/run_id"
  Example: .venv/bin/python export_wandb.py "dlee6018-/alphazero-chess/abc123"

  Find the run path on wandb.ai: click a run → Overview → "Run path" field.
  Or just copy the URL: wandb.ai/dlee6018-/alphazero-chess/runs/abc123
    → run_path = "dlee6018-/alphazero-chess/abc123"
"""
import sys
import wandb
import pandas as pd

def export_run(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)

    print(f"Run: {run.name} ({run.state})")
    print(f"Config: {dict(run.config)}")
    print(f"Summary: {dict(run.summary)}")
    print()

    # --- Export full training history (all scalars logged via wandb.log) ---
    history = run.history(samples=10000)  # up to 10k rows; increase if needed
    history.to_csv("wandb_history.csv", index=False)
    print(f"✓ Exported {len(history)} rows to wandb_history.csv")
    print(f"  Columns: {list(history.columns)}")
    print()

    # --- Export system metrics (CPU, memory, disk, etc.) ---
    system_metrics = run.history(stream="events", samples=10000)
    if not system_metrics.empty:
        system_metrics.to_csv("wandb_system_metrics.csv", index=False)
        print(f"✓ Exported {len(system_metrics)} rows to wandb_system_metrics.csv")
        print(f"  Columns: {list(system_metrics.columns)}")
    else:
        print("  No system metrics found")
    print()

    # --- Basic analysis ---
    print("=" * 60)
    print("QUICK ANALYSIS")
    print("=" * 60)

    if "loss/total" in history.columns:
        print(f"\nTotal Loss:")
        print(f"  Start: {history['loss/total'].iloc[0]:.4f}")
        print(f"  End:   {history['loss/total'].dropna().iloc[-1]:.4f}")
        print(f"  Min:   {history['loss/total'].min():.4f}")

    if "loss/policy" in history.columns:
        print(f"\nPolicy Loss:")
        print(f"  Start: {history['loss/policy'].iloc[0]:.4f}")
        print(f"  End:   {history['loss/policy'].dropna().iloc[-1]:.4f}")

    if "loss/value" in history.columns:
        print(f"\nValue Loss:")
        print(f"  Start: {history['loss/value'].iloc[0]:.4f}")
        print(f"  End:   {history['loss/value'].dropna().iloc[-1]:.4f}")

    if "gradients/total_norm" in history.columns:
        grad_col = history["gradients/total_norm"].dropna()
        print(f"\nGradient Norms:")
        print(f"  Mean: {grad_col.mean():.4f}")
        print(f"  Max:  {grad_col.max():.4f}")
        print(f"  Min:  {grad_col.min():.4f}")

    if "gradients/policy_head_norm" in history.columns:
        print(f"  Policy head mean: {history['gradients/policy_head_norm'].dropna().mean():.4f}")
        print(f"  Value head mean:  {history['gradients/value_head_norm'].dropna().mean():.4f}")

    if "throughput/samples_per_sec" in history.columns:
        tp = history["throughput/samples_per_sec"].dropna()
        print(f"\nThroughput:")
        print(f"  Mean: {tp.mean():.0f} samples/sec")
        print(f"  Min:  {tp.min():.0f} samples/sec")
        print(f"  Max:  {tp.max():.0f} samples/sec")

    if "val/accuracy" in history.columns:
        print(f"\nValidation Accuracy:")
        print(f"  Final: {history['val/accuracy'].dropna().iloc[-1]:.4f}")
        print(f"  Best:  {history['val/accuracy'].max():.4f}")

    if "train/accuracy" in history.columns:
        print(f"\nTraining Accuracy:")
        print(f"  Final: {history['train/accuracy'].dropna().iloc[-1]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_wandb.py <entity/project/run_id>")
        print("Example: python export_wandb.py 'dlee6018-/alphazero-chess/abc123'")
        sys.exit(1)

    export_run(sys.argv[1])
