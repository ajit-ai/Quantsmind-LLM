from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Missing token"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": __import__('datetime').datetime.now().isoformat()})

@app.route('/api/users', methods=['GET'])
@require_auth
def list_users():
    return jsonify({"users": []})

@app.route('/api/users', methods=['POST'])
@require_auth
def create_user():
    data = request.get_json()
    if not data or 'username' not in data:
        return jsonify({"error": "Missing username"}), 400
    return jsonify({"username": data['username'], "created": True}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)