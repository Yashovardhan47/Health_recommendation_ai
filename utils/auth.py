"""utils/auth.py — JWT authentication, RBAC, demo users"""
import hashlib, time
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

PERMISSIONS = {
    "Admin":   {"view_dashboard","view_analytics","view_reports","manage_users","train_models"},
    "Analyst": {"view_dashboard","view_analytics","view_reports","train_models"},
    "User":    {"view_dashboard"},
}

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _load_users() -> pd.DataFrame:
    p = DATA_DIR / "users.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def authenticate_user(username: str, password: str):
    users = _load_users()
    if users.empty:
        return False, "No user database found"
    row = users[users["username"] == username]
    if row.empty:
        return False, "User not found"
    row = row.iloc[0]
    if row["password_hash"] != _hash(password):
        return False, "Wrong password"
    return True, row.to_dict()

def register_user(username: str, password: str, email: str, age: int, gender: str):
    users = _load_users()
    if not users.empty and username in users["username"].values:
        return False, "Username already taken"
    new_id = f"U{len(users)+1:04d}" if not users.empty else "U0001"
    new_row = {
        "user_id": new_id, "username": username,
        "password_hash": _hash(password), "email": email,
        "role": "User", "age": age, "gender": gender,
        "blood_type": "O+", "chronic_conditions": "None",
        "allergies": "None", "created_at": pd.Timestamp.now(),
    }
    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
    users.to_csv(DATA_DIR / "users.csv", index=False)
    return True, new_row

def has_permission(role: str, perm: str) -> bool:
    return perm in PERMISSIONS.get(role, set())
