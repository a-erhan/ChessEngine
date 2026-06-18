"""
Chess Engine API - Flask Backend
Egitilmis modellere erisim icin REST API + Frontend servis
"""
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chess_engine import chess_ai, MODEL_CONFIGS

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ─── FRONTEND SERVE ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ─── HEALTH CHECK ────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Sunucu sağlık kontrolü."""
    return jsonify({
        "status": "ok",
        "models": list(MODEL_CONFIGS.keys()),
        "message": "Chess Engine API çalışıyor! ♟️"
    })


# ─── MODEL BİLGİSİ ───────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def get_models():
    """Mevcut modellerin listesini döner."""
    models = {}
    for key, config in MODEL_CONFIGS.items():
        models[key] = {
            "key": key,
            "name": config["name"],
            "emoji": config["emoji"],
            "description": config["description"]
        }
    return jsonify(models)


# ─── HAMLE AL ────────────────────────────────────────────────────────────────

@app.route("/api/move", methods=["POST"])
def get_move():
    """
    Verilen pozisyon ve model için en iyi hamleyi hesaplar.

    Request body:
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "model": "agresif" | "tedbirli" | "akil"
    }

    Response:
    {
        "uci": "e2e4",
        "from": "e2",
        "to": "e4",
        "promotion": null | "q",
        "value": 0.35,
        "model": "Agresif"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body gerekli"}), 400

        fen = data.get("fen")
        model_key = data.get("model", "agresif")

        if not fen:
            return jsonify({"error": "FEN gerekli"}), 400

        if model_key not in MODEL_CONFIGS:
            return jsonify({
                "error": f"Geçersiz model: {model_key}. Geçerli değerler: {list(MODEL_CONFIGS.keys())}"
            }), 400

        result = chess_ai.get_best_move(fen, model_key)

        if "error" in result:
            return jsonify(result), 200  # Oyun bitti vs. hata

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500


# ─── OYUN DURUMU ─────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["POST"])
def get_status():
    """
    Verilen FEN için oyun durumunu döner.

    Request body: {"fen": "..."}
    """
    try:
        data = request.get_json()
        fen = data.get("fen")
        if not fen:
            return jsonify({"error": "FEN gerekli"}), 400

        status = chess_ai.get_game_status(fen)
        return jsonify(status)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── BAŞLANGIC FEN ───────────────────────────────────────────────────────────

@app.route("/api/start", methods=["GET"])
def get_start():
    """Başlangıç FEN ve oyun bilgilerini döner."""
    return jsonify({
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "models": {k: {"name": v["name"], "emoji": v["emoji"], "description": v["description"]}
                   for k, v in MODEL_CONFIGS.items()}
    })


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Chess Engine API başlatılıyor — port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
