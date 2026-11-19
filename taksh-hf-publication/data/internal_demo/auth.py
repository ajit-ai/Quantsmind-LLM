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