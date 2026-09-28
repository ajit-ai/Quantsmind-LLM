import json
import os
from pathlib import Path

demo_repo = Path("data/internal_demo")
demo_repo.mkdir(parents=True, exist_ok=True)

files = {
    "auth.py": """
import hashlib
import secrets
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self._users = {}

    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hashed.hex()}"

    def verify_password(self, password: str, stored: str) -> bool:
        salt, hashed = stored.split('$')
        new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return new_hash.hex() == hashed

    def create_user(self, username: str, password: str) -> dict:
        if username in self._users:
            raise ValueError(f"User {username} already exists")
        user = {
            "username": username,
            "password_hash": self.hash_password(password),
            "created_at": datetime.now().isoformat(),
        }
        self._users[username] = user
        return {"username": username, "created_at": user["created_at"]}

    def authenticate(self, username: str, password: str) -> bool:
        user = self._users.get(username)
        if not user:
            return False
        return self.verify_password(password, user["password_hash"])
""",
    "api.py": """
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
""",
    "utils.py": """
import re
from typing import Optional

def sanitize_input(text: str, max_length: int = 1000) -> str:
    text = text.strip()[:max_length]
    text = re.sub(r'[<>]', '', text)
    return text

def format_currency(amount: float, currency: str = "USD") -> str:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    symbol = symbols.get(currency, "$")
    return f"{symbol}{amount:,.2f}"

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator
""",
    "config.py": """
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    debug: bool = False
    database_url: str = "sqlite:///app.db"
    secret_key: str = ""
    max_connections: int = 10
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls):
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            database_url=os.getenv("DATABASE_URL", "sqlite:///app.db"),
            secret_key=os.getenv("SECRET_KEY", ""),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "10")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30")),
        )

config = AppConfig.from_env()
""",
    "models.py": """
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid

@dataclass
class User:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    owner_id: str = ""
    members: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    title: str = ""
    status: str = "todo"
    assignee_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
""",
}

for filename, content in files.items():
    (demo_repo / filename).write_text(content.strip(), encoding="utf-8")

print(f"Created {len(files)} demo files in {demo_repo}")
