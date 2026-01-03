# Full (minimal but complete) AlphaZero-style self-play + training loop for chess.
# - Uses your (PUCT) MCTS with a neural policy/value network.
# - Collects (state, pi, z) from self-play into a replay buffer.
# - Trains the network on batches from the replay buffer.
#
# Assumptions:
# 1) You have python-chess installed: pip install python-chess
# 2) You have PyTorch installed.
#
# IMPORTANT DESIGN CHOICE (move encoding):
# - To keep this robust and simple, we use a fixed action space of size 64*64*5 = 20480.
#   index = from_square * 64 * 5 + to_square * 5 + promo_id
#   promo_id: 0 = no promo, 1 = n, 2 = b, 3 = r, 4 = q
# - Your network's policy head must output logits of shape (B, 20480).

import math
import random
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Tuple, Optional, List

import numpy as np
import chess

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Board encoder: (18,8,8)
# -----------------------------
def board_to_input_planes(board: chess.Board) -> np.ndarray:
    planes: List[np.ndarray] = []

    piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for color in (chess.WHITE, chess.BLACK):
        for p in piece_order:
            plane = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(p, color):
                r = 7 - chess.square_rank(sq)
                f = chess.square_file(sq)
                plane[r, f] = 1.0
            planes.append(plane)

    # side to move
    planes.append(np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32))

    # castling rights (WK, WQ, BK, BQ)
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for r in rights:
        planes.append(np.ones((8, 8), dtype=np.float32) if r else np.zeros((8, 8), dtype=np.float32))

    # en passant
    ep = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        rr = 7 - chess.square_rank(board.ep_square)
        ff = chess.square_file(board.ep_square)
        ep[rr, ff] = 1.0
    planes.append(ep)

    x = np.stack(planes, axis=0)  # (18, 8, 8)
    return x


# -----------------------------
# Move encoding: 20480 actions
# -----------------------------
class MoveIndexer:
    """
    Fixed action space: 64*64*5 = 20480
    promo_id: 0 none, 1 n, 2 b, 3 r, 4 q
    """
    PROMO_TO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    ID_TO_PROMO = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}

    @staticmethod
    def n_actions() -> int:
        return 64 * 64 * 5

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        promo = move.promotion
        promo_id = MoveIndexer.PROMO_TO_ID.get(promo, 0)
        return (move.from_square * 64 * 5) + (move.to_square * 5) + promo_id

    @staticmethod
    def legal_indices(board: chess.Board) -> List[int]:
        return [MoveIndexer.move_to_index(m) for m in board.legal_moves]


# -----------------------------
# Model (same spirit as before, but policy head outputs 20480 logits)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 18, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PolicyHead(nn.Module):
    def __init__(self, channels: int = 64, n_actions: int = 20480):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * 8 * 8, n_actions)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)  # logits


class ValueHead(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        v = torch.tanh(self.fc2(x))
        return v


class AlphaZeroChessNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 10, n_actions: int = 20480, value_hidden: int = 256):
        super().__init__()
        self.stem = ConvBlock(18, channels)
        self.residuals = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        self.policy = PolicyHead(channels, n_actions)
        self.value = ValueHead(channels, value_hidden)

    def forward(self, x):
        x = self.stem(x)
        for b in self.residuals:
            x = b(x)
        p_logits = self.policy(x)
        v = self.value(x)
        return p_logits, v


# -----------------------------
# MCTS (PUCT) using priors+value from policy_value_fn(board)
# -----------------------------
@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    parent_move: Optional[chess.Move] = None
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    P: Dict[chess.Move, float] = field(default_factory=dict)
    is_expanded: bool = False
    N: int = 0
    W: float = 0.0
    Q: float = 0.0


class MCTS:
    def __init__(self, policy_value_fn, c_puct: float = 1.5, n_simulations: int = 200):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, root_board: chess.Board) -> Tuple[Optional[chess.Move], Dict[chess.Move, float]]:
        root = MCTSNode(root_board.copy(stack=False))
        for _ in range(self.n_simulations):
            leaf, path = self._select(root)
            value = self._expand_and_evaluate(leaf)
            self._backpropagate(path, value)

        if not root.children:
            legal_moves = list(root_board.legal_moves)
            move = random.choice(legal_moves) if legal_moves else None
            return move, {}

        move_visits = {m: child.N for m, child in root.children.items()}
        total = sum(move_visits.values())
        if total <= 0:
            legal_moves = list(root_board.legal_moves)
            move = random.choice(legal_moves) if legal_moves else None
            return move, {}

        probs = {m: n / total for m, n in move_visits.items()}
        best_move = max(move_visits.items(), key=lambda kv: kv[1])[0]
        return best_move, probs

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        node = root
        path = [node]
        while node.is_expanded and (not node.board.is_game_over()):
            if not node.children:
                break
            _, node = self._select_child(node)
            path.append(node)
        return node, path

    def _select_child(self, node: MCTSNode) -> Tuple[chess.Move, MCTSNode]:
        N_sum = sum(ch.N for ch in node.children.values()) + 1e-8
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            Q = child.Q
            P = node.P.get(move, 0.0)
            U = self.c_puct * P * (math.sqrt(N_sum) / (1.0 + child.N))
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over():
            res = board.result()
            if res == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            if res == "0-1":
                return 1.0 if board.turn == chess.BLACK else -1.0
            return 0.0

        priors, value = self.policy_value_fn(board)

        node.is_expanded = True
        node.P = {}
        node.children = {}

        for move in board.legal_moves:
            p = float(priors.get(move, 0.0))
            node.P[move] = p
            b2 = board.copy(stack=False)
            b2.push(move)
            node.children[move] = MCTSNode(b2, parent=node, parent_move=move)

        s = sum(node.P.values())
        if s > 0:
            inv = 1.0 / s
            for m in node.P:
                node.P[m] *= inv
        else:
            legal = list(node.children.keys())
            if legal:
                u = 1.0 / len(legal)
                for m in legal:
                    node.P[m] = u

        return float(value)

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value


# -----------------------------
# policy_value_fn wrapper around model
# -----------------------------
@torch.no_grad()
def make_policy_value_fn(model: nn.Module, device: torch.device, add_root_dirichlet: bool = False,
                         dir_alpha: float = 0.3, dir_eps: float = 0.25):
    """
    Returns a function(board) -> (priors: Dict[Move,float], value: float)

    - Priors are computed from model policy logits over the 20480 fixed action space,
      then masked to legal moves and softmaxed.
    - Optional Dirichlet noise can be injected (typically for self-play at root only).
      If you want "root-only", apply noise outside or call this with add_root_dirichlet=True
      only for the root evaluation in your self-play loop.
    """
    n_actions = MoveIndexer.n_actions()

    def pv(board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        x_np = board_to_input_planes(board)  # (18,8,8)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,18,8,8)

        p_logits, v = model(x)
        p_logits = p_logits.squeeze(0)  # (n_actions,)
        v = float(v.squeeze().item())

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, v

        legal_idx = torch.tensor([MoveIndexer.move_to_index(m) for m in legal_moves], device=device, dtype=torch.long)

        # mask to legal indices and softmax over them
        legal_logits = p_logits.index_select(0, legal_idx)
        legal_probs = torch.softmax(legal_logits, dim=0)

        if add_root_dirichlet:
            # Dirichlet noise over legal moves
            noise = np.random.dirichlet([dir_alpha] * len(legal_moves)).astype(np.float32)
            noise_t = torch.from_numpy(noise).to(device)
            legal_probs = (1 - dir_eps) * legal_probs + dir_eps * noise_t
            legal_probs = legal_probs / legal_probs.sum()

        priors = {m: float(legal_probs[i].item()) for i, m in enumerate(legal_moves)}
        return priors, v

    return pv


# -----------------------------
# Replay Buffer
# -----------------------------
@dataclass
class Sample:
    state: np.ndarray   # (18,8,8) float32
    pi: np.ndarray      # (n_actions,) float32
    z: float            # scalar in [-1,1]


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def add_game(self, samples: List[Sample]):
        self.buf.extend(samples)

    def __len__(self):
        return len(self.buf)

    def sample_batch(self, batch_size: int) -> List[Sample]:
        return random.sample(self.buf, batch_size)


# -----------------------------
# Self-play: generate (state, pi, z)
# -----------------------------
def choose_move_from_probs(move_probs: Dict[chess.Move, float], temperature: float) -> chess.Move:
    moves = list(move_probs.keys())
    probs = np.array([move_probs[m] for m in moves], dtype=np.float64)

    if temperature <= 1e-6:
        return moves[int(np.argmax(probs))]

    # temperature scaling: p_i^(1/T)
    probs = np.power(probs, 1.0 / temperature)
    probs = probs / probs.sum()
    return np.random.choice(moves, p=probs)


def play_self_play_game(model: nn.Module,
                        device: torch.device,
                        n_simulations: int = 200,
                        c_puct: float = 1.5,
                        max_moves: int = 300,
                        temp_moves: int = 20,
                        temperature: float = 1.0,
                        dir_alpha: float = 0.3,
                        dir_eps: float = 0.25) -> List[Sample]:
    """
    Plays one self-play game and returns training samples (state, pi, z).

    - For first 'temp_moves' plies, sample moves with given temperature.
    - After that, play argmax (temperature ~ 0).
    - Dirichlet noise applied at the root priors for exploration.
    """
    board = chess.Board()

    # store (state, pi, player_sign) per ply; player_sign is +1 if side-to-move is White else -1
    traj = []

    model.eval()

    for ply in range(max_moves):
        if board.is_game_over():
            break

        # Root evaluation with Dirichlet noise (root-only)
        pv_root = make_policy_value_fn(model, device, add_root_dirichlet=True, dir_alpha=dir_alpha, dir_eps=dir_eps)
        pv_no_noise = make_policy_value_fn(model, device, add_root_dirichlet=False)

        # MCTS uses a policy_value_fn; we want noise only at the root.
        # Easiest: create MCTS with a wrapper that uses noisy pv for the first expansion encountered (root),
        # but since our MCTS builds a fresh tree each move, we can just pass the noisy pv for this move.
        mcts = MCTS(policy_value_fn=pv_root, c_puct=c_puct, n_simulations=n_simulations)

        best_move, move_probs = mcts.search(board)
        if best_move is None or not move_probs:
            # fallback: random legal
            legal = list(board.legal_moves)
            if not legal:
                break
            best_move = random.choice(legal)
            move_probs = {m: 1.0 / len(legal) for m in legal}

        # Build full pi vector over fixed action space
        n_actions = MoveIndexer.n_actions()
        pi = np.zeros((n_actions,), dtype=np.float32)
        for m, p in move_probs.items():
            pi[MoveIndexer.move_to_index(m)] = float(p)

        state = board_to_input_planes(board)  # (18,8,8)
        player_sign = 1.0 if board.turn == chess.WHITE else -1.0
        traj.append((state, pi, player_sign))

        # choose move (explore early, exploit later)
        T = temperature if ply < temp_moves else 1e-8
        move = choose_move_from_probs(move_probs, temperature=T)
        board.push(move)

    # Determine game outcome from White's perspective
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            outcome_white = 1.0
        elif res == "0-1":
            outcome_white = -1.0
        else:
            outcome_white = 0.0
    else:
        # If we hit max_moves, treat as draw (common pragmatic choice)
        outcome_white = 0.0

    # Convert to samples: z from perspective of player-to-move at that state
    samples: List[Sample] = []
    for state, pi, player_sign in traj:
        # player_sign = +1 means side-to-move is White at that state, so z = outcome_white
        # player_sign = -1 means side-to-move is Black at that state, so z = -outcome_white
        z = outcome_white * player_sign
        samples.append(Sample(state=state, pi=pi, z=float(z)))

    return samples


# -----------------------------
# Training step
# -----------------------------
def train_on_batch(model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                   batch: List[Sample], value_weight: float = 1.0, policy_weight: float = 1.0) -> Dict[str, float]:
    model.train()

    states = torch.from_numpy(np.stack([s.state for s in batch], axis=0)).to(device)  # (B,18,8,8)
    target_pi = torch.from_numpy(np.stack([s.pi for s in batch], axis=0)).to(device)  # (B,A)
    target_z = torch.tensor([s.z for s in batch], dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)

    logits, v = model(states)  # logits: (B,A), v: (B,1)

    # Policy loss: cross-entropy with target distribution pi (MCTS visit distribution)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(v, target_z)

    loss = policy_weight * policy_loss + value_weight * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "v_mean": float(v.mean().item()),
    }


# -----------------------------
# Full training loop
# -----------------------------
def train_loop(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    channels: int = 64,
    n_blocks: int = 10,
    value_hidden: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    iterations: int = 50,
    games_per_iter: int = 10,
    n_simulations: int = 200,
    c_puct: float = 1.5,
    max_moves: int = 300,
    temp_moves: int = 20,
    temperature: float = 1.0,
    dir_alpha: float = 0.3,
    dir_eps: float = 0.25,
    replay_capacity: int = 200_000,
    batch_size: int = 256,
    train_steps_per_iter: int = 200,
    min_buffer_to_train: int = 5_000,
    save_path: str = "az_chess.pt",
):
    device_t = torch.device(device)
    n_actions = MoveIndexer.n_actions()

    model = AlphaZeroChessNet(channels=channels, n_blocks=n_blocks, n_actions=n_actions, value_hidden=value_hidden).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    buffer = ReplayBuffer(capacity=replay_capacity)

    for it in range(1, iterations + 1):
        # --- Self-play data generation ---
        all_new = 0
        for g in range(games_per_iter):
            samples = play_self_play_game(
                model=model,
                device=device_t,
                n_simulations=n_simulations,
                c_puct=c_puct,
                max_moves=max_moves,
                temp_moves=temp_moves,
                temperature=temperature,
                dir_alpha=dir_alpha,
                dir_eps=dir_eps,
            )
            buffer.add_game(samples)
            all_new += len(samples)

        print(f"[iter {it}] self-play added {all_new} samples, buffer size={len(buffer)}")

        # --- Training ---
        if len(buffer) < min_buffer_to_train:
            print(f"[iter {it}] buffer < {min_buffer_to_train}, skipping training")
        else:
            metrics_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "v_mean": 0.0}
            for step in range(train_steps_per_iter):
                batch = buffer.sample_batch(batch_size)
                metrics = train_on_batch(model, optimizer, device_t, batch)
                for k in metrics_acc:
                    metrics_acc[k] += metrics[k]
            for k in metrics_acc:
                metrics_acc[k] /= train_steps_per_iter
            print(f"[iter {it}] train: {metrics_acc}")

        # --- Save checkpoint ---
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "channels": channels,
                    "n_blocks": n_blocks,
                    "value_hidden": value_hidden,
                    "n_actions": n_actions,
                },
                "iteration": it,
            },
            save_path,
        )
        print(f"[iter {it}] saved to {save_path}")

    return model


if __name__ == "__main__":
    # Start a run with small defaults. Scale up gradually.
    train_loop(
        iterations=10,
        games_per_iter=4,
        n_simulations=100,      # start smaller for speed; increase later
        train_steps_per_iter=100,
        batch_size=128,
        min_buffer_to_train=1000,
    )
