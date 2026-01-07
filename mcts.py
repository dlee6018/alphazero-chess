import math
import random
import chess
import numpy as np
from board import board_to_input_planes
from typing import Dict, Optional, Callable, Tuple, List
import uuid

class AlphaZeroMoveIndexer:
    """
    Class to encode and decode chess moves
    """
    def __init__(self):
        self.queen_dirs = [
            (0, 1),   # North
            (1, 1),   # North-East
            (1, 0),   # East
            (1, -1),  # South-East
            (0, -1),  # South
            (-1, -1), # South-West
            (-1, 0),  # West
            (-1, 1)   # North-West
        ]
        
        self.knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        
        # Underpromotion pieces map (Index -> Piece)
        self.under_promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        
        # Underpromotion direction map (Index -> File Delta)
        # 0=Left(-1), 1=Forward(0), 2=Right(+1)
        self.under_promo_file_deltas = [-1, 0, 1]

        self.move_to_plane = {}
        
        count = 0
        # Planes 0 - 55 queen like moves - covers rook, bishop, queen, king, pawn
        for dx, dy in self.queen_dirs:
            for dist in range(1, 8):
                self.move_to_plane[(dx * dist, dy * dist)] = count
                count += 1

        # Planes 56 - 63 knight like moves
        for dx, dy in self.knight_moves:
            self.move_to_plane[(dx, dy)] = count
            count += 1
            
        

    def encode(self, move: chess.Move) -> int:
        """
        Encodes a chess move into a plane index
        Total of 64 squares * 73 planes = 4,672 possible moves
        """
        from_sq = move.from_square
        to_sq = move.to_square
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
        
        # Planes 64 - 72 underpromotions to knight, bishop, rook
        # 3 pieces (knight, bishop, rook) and 3 ways to get promote - (upleft, up, upright)
        if move.promotion and move.promotion != chess.QUEEN:
            p_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
            d_idx = { -1: 0, 0: 1, 1: 2 }[df]
            plane_idx = 64 + (d_idx * 3) + p_idx
        else:
            # fetch from the pre-processed planes during init
            plane_idx = self.move_to_plane.get((df, dr))
            if plane_idx is None: return None
            
        return (from_sq * 73) + plane_idx 

class MCTSNode:
    """
    Class to represent a node in the MCTS tree
    """
    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"]=None, parent_move: Optional[chess.Move]=None):
        self.board = board
        self.parent = parent
        self.parent_move = parent_move
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

        # Node statistics
        self.N = 0          # visit count of this node
        self.W = 0.0        # total value 
        self.Q = 0.0        # self.W / self.N

        # Edge priors P(s,a): stored in parent.P[move]
        self.P: Dict[chess.Move, float] = {}  # only meaningful for nodes with children

class MCTS:
    """
    Class to perform MCTS search
    """
    def __init__(self, c_puct: float = 1.5, n_simulations: int = 200, batch_size: int = 1,
                 inference_queue=None, inference_result_queue=None, worker_id: int = 0):
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.batch_size = batch_size
        self.move_indexer = AlphaZeroMoveIndexer()
        self.inference_queue = inference_queue
        self.response_queue = inference_result_queue  # dedicated Queue per worker
        self.worker_id = worker_id
        self.pending_inferences = {}  # request_id -> (pending_leaves, batch_tensor)

    def search(self, root_board: chess.Board) -> Tuple[Optional[chess.Move], Dict[chess.Move, float]]:
        root = MCTSNode(root_board.copy(stack=False)) 

        pending_leaves = []  # List of (leaf, path) tuples for non-terminal nodes

        for sim in range(self.n_simulations):
            leaf, path = self._select(root)

            if leaf.board.is_game_over():
                value = self._get_terminal_value(leaf.board)
                self._backpropagate(path, value)
                continue

            pending_leaves.append((leaf, path))

            # Evaluate batch when full or at end of simulations
            if len(pending_leaves) >= self.batch_size or sim == self.n_simulations - 1:
                if pending_leaves:
                    self._batch_evaluate_and_backpropagate(pending_leaves)
                    pending_leaves = []

        if not root.children:
            legal_moves = list(root_board.legal_moves)
            move = random.choice(legal_moves) if legal_moves else None
            return move, {}

        move_visits = {move: child.N for move, child in root.children.items()}
        total_visits = sum(move_visits.values())

        if total_visits == 0:
            legal_moves = list(root_board.legal_moves)
            move = random.choice(legal_moves) if legal_moves else None
            return move, {}

        move_probs = {m: n / total_visits for m, n in move_visits.items()}
        
        # Sample move based on probability distribution instead of always picking the best
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        sampled_move = random.choices(moves, weights=probs, k=1)[0]
        
        return sampled_move, move_probs

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Returns a leaf node and the path from root to leaf.
        """
        node = root
        path = [node]

        while node.is_expanded and (not node.board.is_game_over()):
            if not node.children:
                break
            _, node = self._select_child(node)
            path.append(node)

        return node, path

    def _select_child(self, node: MCTSNode) -> Tuple[chess.Move, MCTSNode]:
        assert node.children, "Tried to select a child from a node with no children."

        N_sum = sum(child.N for child in node.children.values()) + 1e-8 # add small epsilon to avoid division by zero

        best_score = -float("inf")
        best_move: Optional[chess.Move] = None
        best_child: Optional[MCTSNode] = None

        for move, child in node.children.items():
            Q = child.Q
            assert node.P, "Node must have P values set (should be expanded)"
            P = node.P[move] # prior probability of the move from the neural net
            U = self.c_puct * P * (math.sqrt(N_sum) / (1.0 + child.N))
            score = Q + U

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        # Should always be set given non-empty children
        assert best_move is not None and best_child is not None
        
        return best_move, best_child

    def _get_terminal_value(self, board: chess.Board) -> float:
        """Get value for terminal board position."""
        result = board.result()  # e.g: "1-0", "0-1", "1/2-1/2"
        if result == "1-0":
            return 1.0 if board.turn == chess.WHITE else -1.0
        if result == "0-1":
            return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0

    def _batch_evaluate_and_backpropagate(self, pending_leaves: List[Tuple[MCTSNode, List[MCTSNode]]]) -> None:
        """Evaluate a batch of leaves and backpropagate results."""
        if not pending_leaves:
            return

        boards = [leaf.board for leaf, _ in pending_leaves]
        board_tensors = []
        for board in boards:
            tensor = board_to_input_planes(board)
            board_tensors.append(tensor)

        # Stack into batch: (batch_size, 18, 8, 8)
        batch_tensor = np.stack(board_tensors)

        # Send inference request with worker_id for routing response
        request_id = str(uuid.uuid4())
        self.inference_queue.put((self.worker_id, batch_tensor, request_id))

        # Block on dedicated response queue (no polling), each process has it's unique queue
        response_request_id, policy_logits_batch, value_batch = self.response_queue.get()

        assert response_request_id == request_id, f"Request ID mismatch: {response_request_id} != {request_id}"
        
        for i, (leaf, path) in enumerate(pending_leaves):
            board = leaf.board
            policy_logits = policy_logits_batch[i]  # (4672,)
            value = float(value_batch[i].squeeze().item())
            
            legal_moves = list(board.legal_moves)
            legal_moves_dict = {}
            for move in legal_moves:
                move_idx = self.move_indexer.encode(move) # encode each move to a number 
                assert move_idx is not None, "Move index is None"
                
                logit_value = float(policy_logits[move_idx].item()) # get the policy logit for the move
                legal_moves_dict[move] = logit_value
            
            # Expand node
            leaf.is_expanded = True
            leaf.P = {}
            leaf.children = {}
            
            # Build children for all legal moves
            for move in board.legal_moves:
                p = float(legal_moves_dict.get(move, 0.0))
                leaf.P[move] = p
                
                child_board = board.copy(stack=False)
                child_board.push(move)
                leaf.children[move] = MCTSNode(child_board, parent=leaf, parent_move=move)
            
            # Renormalize priors over legal moves, fallback to uniform if all 0
            s = sum(leaf.P.values())
            if s > 0.0:
                inv = 1.0 / s
                for m in leaf.P:
                    leaf.P[m] *= inv
            else:
                legal = list(leaf.children.keys())
                if legal:
                    uniform = 1.0 / len(legal)
                    for m in legal:
                        leaf.P[m] = uniform
            
            # Add Dirichlet noise to root node only (for exploration)
            if leaf.parent is None:  
                dirichlet_alpha = 0.3  # Typical value from AlphaZero
                epsilon = 0.25  # Mixing parameter
                
                legal_moves_list = list(leaf.P.keys())
                num_moves = len(legal_moves_list)
                
                if num_moves > 0:
                    # Sample Dirichlet noise
                    dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * num_moves)
                    
                    # Mix priors with noise
                    for i, move in enumerate(legal_moves_list):
                        leaf.P[move] = (1 - epsilon) * leaf.P[move] + epsilon * dirichlet_noise[i]
                    
                    # Renormalize after adding noise
                    s = sum(leaf.P.values())
                    if s > 0.0:
                        inv = 1.0 / s
                        for m in leaf.P:
                            leaf.P[m] *= inv
            
            self._backpropagate(path, value)

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # flip perspective when moving up one ply
    



if __name__ == "__main__":
    b = chess.Board()
    mcts = MCTS(c_puct=1.5, n_simulations=300)
    move, probs = mcts.search(b)
    print("Best move:", move)