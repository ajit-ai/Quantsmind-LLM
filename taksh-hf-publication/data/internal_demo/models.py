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