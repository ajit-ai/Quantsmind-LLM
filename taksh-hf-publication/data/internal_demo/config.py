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