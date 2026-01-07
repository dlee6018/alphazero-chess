import torch
import chess
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from model import AlphaZeroChessNet
from board import board_to_input_planes
from mcts import AlphaZeroMoveIndexer
import sys
import os

app = Flask(__name__)

# Global state
model = None
move_indexer = None
device = None
board = chess.Board()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AlphaZero Chess - Inference</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1e1e1e;
            color: #fff;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .left-panel {
            flex: 1;
        }
        .right-panel {
            width: 300px;
        }
        .controls {
            margin-bottom: 20px;
            padding: 15px;
            background: #2d2d2d;
            border-radius: 8px;
        }
        .controls button {
            margin: 5px;
            padding: 10px 15px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .controls button:hover {
            background: #357abd;
        }
        .controls input[type="file"] {
            margin: 5px;
            padding: 8px;
            background: #3d3d3d;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
        }
        .status {
            padding: 10px;
            background: #2d2d2d;
            border-radius: 4px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        #chessboard {
            display: grid;
            grid-template-columns: repeat(8, 60px);
            grid-template-rows: repeat(8, 60px);
            gap: 0;
            border: 3px solid #555;
            width: 480px;
            height: 480px;
            margin: 20px 0;
        }
        .square {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            cursor: pointer;
            position: relative;
        }
        .square.light {
            background: #f0d9b5;
        }
        .square.dark {
            background: #b58863;
        }
        .square.selected {
            background: #ff6b6b !important;
        }
        .square.last-move {
            background: #cdd26a !important;
        }
        .square:hover {
            opacity: 0.8;
        }
        .file-label {
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            color: #ccc;
        }
        .rank-label {
            position: absolute;
            left: -25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 12px;
            color: #ccc;
        }
        .board-container {
            position: relative;
            display: inline-block;
        }
        .info-panel {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .info-panel h3 {
            margin-top: 0;
        }
        #moveHistory {
            background: #1e1e1e;
            color: #fff;
            padding: 10px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .move-entry {
            margin: 5px 0;
            padding: 5px;
            background: #3d3d3d;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>AlphaZero Chess - Inference</h1>
    
    <div class="controls">
        <input type="file" id="checkpointFile" accept=".pt" />
        <button onclick="loadCheckpoint()">Load Checkpoint</button>
        <button onclick="resetGame()">Reset Game</button>
        <button onclick="undoMove()">Undo Move</button>
        <span id="modelStatus" style="margin-left: 20px; color: #888;">Loading checkpoint_8000.pt...</span>
    </div>
    
    <div class="container">
        <div class="left-panel">
            <div class="status" id="status">Loading model...</div>
            <div class="board-container">
                <div id="chessboard"></div>
            </div>
        </div>
        
        <div class="right-panel">
            <div class="info-panel">
                <h3>Model Info</h3>
                <div id="modelInfo">Not loaded</div>
                <div id="valueInfo" style="margin-top: 10px;">Value: -</div>
            </div>
            
            <div class="info-panel">
                <h3>Move History</h3>
                <div id="moveHistory"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedSquare = null;
        let lastMoveFrom = null;
        let lastMoveTo = null;
        
        const pieces = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        };
        
        function loadCheckpoint() {
            const fileInput = document.getElementById('checkpointFile');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a checkpoint file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/load_checkpoint', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('modelStatus').textContent = 'Model loaded: ' + file.name;
                    document.getElementById('modelInfo').textContent = 'Model: ' + file.name;
                    updateBoard();
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Error loading checkpoint: ' + error);
            });
        }
        
        function resetGame() {
            fetch('/reset', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    selectedSquare = null;
                    lastMoveFrom = null;
                    lastMoveTo = null;
                    updateBoard();
                    updateMoveHistory([]);
                    document.getElementById('valueInfo').textContent = 'Value: -';
                });
        }
        
        function undoMove() {
            fetch('/undo', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateBoard();
                    }
                });
        }
        
        function updateBoard() {
            fetch('/get_board')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('status').textContent = data.error;
                        return;
                    }
                    
                    const board = data.board;
                    const turn = data.turn;
                    const game_over = data.game_over;
                    const result = data.result;
                    
                    // Update status
                    if (game_over) {
                        if (result === '1-0') {
                            document.getElementById('status').textContent = 'White wins!';
                        } else if (result === '0-1') {
                            document.getElementById('status').textContent = 'Black wins!';
                        } else {
                            document.getElementById('status').textContent = 'Draw!';
                        }
                    } else {
                        document.getElementById('status').textContent = (turn === 'white' ? 'White' : 'Black') + ' to move';
                    }
                    
                    // Draw board
                    const boardDiv = document.getElementById('chessboard');
                    boardDiv.innerHTML = '';
                    
                    for (let rank = 7; rank >= 0; rank--) {
                        for (let file = 0; file < 8; file++) {
                            const square = rank * 8 + file;
                            const squareDiv = document.createElement('div');
                            squareDiv.className = 'square ' + ((rank + file) % 2 === 0 ? 'light' : 'dark');
                            squareDiv.dataset.square = square;
                            
                            // Highlight selected square
                            if (selectedSquare === square) {
                                squareDiv.classList.add('selected');
                            }
                            
                            // Highlight last move
                            if (lastMoveFrom === square || lastMoveTo === square) {
                                squareDiv.classList.add('last-move');
                            }
                            
                            // Add piece
                            const piece = board[square];
                            if (piece) {
                                squareDiv.textContent = pieces[piece] || piece;
                            }
                            
                            // Add labels
                            if (rank === 0) {
                                const fileLabel = document.createElement('div');
                                fileLabel.className = 'file-label';
                                fileLabel.textContent = String.fromCharCode(97 + file);
                                squareDiv.appendChild(fileLabel);
                            }
                            if (file === 0) {
                                const rankLabel = document.createElement('div');
                                rankLabel.className = 'rank-label';
                                rankLabel.textContent = (rank + 1).toString();
                                squareDiv.appendChild(rankLabel);
                            }
                            
                            squareDiv.onclick = () => handleSquareClick(square);
                            boardDiv.appendChild(squareDiv);
                        }
                    }
                    
                    updateMoveHistory(data.move_history || []);
                });
        }
        
        function handleSquareClick(square) {
            if (selectedSquare === null) {
                // Select square
                fetch('/get_board')
                    .then(response => response.json())
                    .then(data => {
                        const board = data.board;
                        const piece = board[square];
                        if (piece) {
                            const isWhitePiece = piece === piece.toUpperCase();
                            const isBlackPiece = piece === piece.toLowerCase();
                            if ((isWhitePiece && data.turn === 'white') || (isBlackPiece && data.turn === 'black')) {
                                selectedSquare = square;
                                updateBoard();
                            }
                        }
                    });
            } else {
                // Make move
                const fromSquare = selectedSquare;
                const toSquare = square;
                
                fetch('/make_move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ from: fromSquare, to: toSquare })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        selectedSquare = null;
                        lastMoveFrom = fromSquare;
                        lastMoveTo = toSquare;
                        updateBoard();
                        
                        if (data.model_move) {
                            setTimeout(() => {
                                lastMoveFrom = data.model_move.from;
                                lastMoveTo = data.model_move.to;
                                document.getElementById('valueInfo').textContent = 'Value: ' + data.value.toFixed(4);
                                updateBoard();
                            }, 500);
                        }
                    } else {
                        alert('Invalid move: ' + data.error);
                        selectedSquare = null;
                        updateBoard();
                    }
                });
            }
        }
        
        function updateMoveHistory(history) {
            const historyDiv = document.getElementById('moveHistory');
            historyDiv.innerHTML = '';
            history.forEach(entry => {
                const div = document.createElement('div');
                div.className = 'move-entry';
                div.textContent = entry;
                historyDiv.appendChild(div);
            });
            historyDiv.scrollTop = historyDiv.scrollHeight;
        }
        
        // Auto-load checkpoint_8000.pt on page load (best checkpoint: lowest total loss 3.5313)
        window.addEventListener('load', function() {
            fetch('/load_checkpoint_path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: 'checkpoint_8000.pt' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('modelStatus').textContent = 'Model loaded: ' + data.filename;
                    document.getElementById('modelInfo').textContent = 'Model: ' + data.filename;
                    updateBoard();
                } else {
                    document.getElementById('modelStatus').textContent = 'Auto-load failed: ' + data.error;
                }
            })
            .catch(error => {
                document.getElementById('modelStatus').textContent = 'Auto-load failed: ' + error;
            });
        });
        
        // Initial board update
        updateBoard();
        setInterval(updateBoard, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def load_checkpoint_from_path(checkpoint_path):
    """Helper function to load checkpoint from file path"""
    global model, move_indexer, device, board
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlphaZeroChessNet(channels=256, n_blocks=20, n_moves=4672).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        move_indexer = AlphaZeroMoveIndexer()
        board = chess.Board()
        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    global model, move_indexer, device, board
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Save uploaded file temporarily
        checkpoint_path = f'/tmp/{file.filename}'
        file.save(checkpoint_path)
        
        success, error = load_checkpoint_from_path(checkpoint_path)
        
        # Clean up temp file
        os.remove(checkpoint_path)
        
        if success:
            return jsonify({'success': True, 'filename': file.filename})
        else:
            return jsonify({'success': False, 'error': error})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_checkpoint_path', methods=['POST'])
def load_checkpoint_path():
    """Load checkpoint from a file path on the server"""
    data = request.json
    checkpoint_path = data.get('path', 'checkpoint_7500.pt')
    
    success, error = load_checkpoint_from_path(checkpoint_path)
    
    if success:
        return jsonify({'success': True, 'filename': os.path.basename(checkpoint_path)})
    else:
        return jsonify({'success': False, 'error': error})

@app.route('/reset', methods=['POST'])
def reset():
    global board
    board = chess.Board()
    return jsonify({'success': True})

@app.route('/undo', methods=['POST'])
def undo():
    global board
    if len(board.move_stack) >= 2:
        board.pop()
        board.pop()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Not enough moves to undo'})

@app.route('/get_board')
def get_board():
    global model, board
    
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    # Convert board to string representation
    board_str = str(board)
    board_dict = {}
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                board_dict[square] = piece.symbol()
            else:
                board_dict[square] = None
    
    # Get move history
    move_history = []
    temp_board = chess.Board()
    for i, move in enumerate(board.move_stack):
        if i % 2 == 0:
            move_history.append(f"Move {i//2 + 1}: {move.uci()}")
        else:
            move_history[-1] += f" - {move.uci()}"
    
    return jsonify({
        'board': board_dict,
        'turn': 'white' if board.turn == chess.WHITE else 'black',
        'game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else None,
        'move_history': move_history
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    global model, board, move_indexer, device
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    if board.is_game_over():
        return jsonify({'success': False, 'error': 'Game is over'})
    
    data = request.json
    from_square = data['from']
    to_square = data['to']
    
    move = chess.Move(from_square, to_square)
    
    # Check for promotion
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        if (chess.square_rank(to_square) == 7 and board.turn == chess.WHITE) or \
           (chess.square_rank(to_square) == 0 and board.turn == chess.BLACK):
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    
    if move not in board.legal_moves:
        return jsonify({'success': False, 'error': 'Illegal move'})
    
    board.push(move)
    
    response = {'success': True}
    
    if not board.is_game_over():
        # Get model move
        model_move, value = get_model_move()
        if model_move:
            board.push(model_move)
            response['model_move'] = {
                'from': model_move.from_square,
                'to': model_move.to_square,
                'uci': model_move.uci()
            }
            response['value'] = value
        else:
            response['error'] = 'Model could not find a legal move'
    
    return jsonify(response)

@torch.no_grad()
def get_model_move():
    """Get model's move for the current board position"""
    global model, board, move_indexer, device
    
    input_planes = board_to_input_planes(board)
    input_tensor = torch.from_numpy(input_planes).unsqueeze(0).to(device)
    
    policy_logits, value = model(input_tensor)
    policy_logits = policy_logits.squeeze(0).cpu().numpy()
    value = value.item()
    
    legal_moves = list(board.legal_moves)
    legal_indices = []
    for move in legal_moves:
        idx = move_indexer.encode(move)
        if idx is not None and 0 <= idx < 4672:
            legal_indices.append((idx, move))
    
    if not legal_indices:
        return None, value
    
    legal_probs = []
    for idx, move in legal_indices:
        legal_probs.append((policy_logits[idx], move))
    
    best_move = max(legal_probs, key=lambda x: x[0])[1]
    return best_move, value

if __name__ == '__main__':
    port = 5000
    checkpoint_path = 'checkpoint_8000.pt'  # Best checkpoint: lowest total loss (3.5313)
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            checkpoint_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        checkpoint_path = sys.argv[2]
    
    print(f"Starting web server on port {port}")
    
    # Try to load checkpoint automatically on startup
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        success, error = load_checkpoint_from_path(checkpoint_path)
        if success:
            print(f"✓ Successfully loaded {checkpoint_path}")
        else:
            print(f"✗ Failed to load checkpoint: {error}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path} (will try to load on page load)")
    
    print(f"Open http://localhost:{port} in your browser")
    print("Or if on remote server, use: http://<server-ip>:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
