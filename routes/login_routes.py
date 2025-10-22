from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse, FileResponse 
from itsdangerous import URLSafeTimedSerializer
import bcrypt
from services.database_service import db_service
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

# Secret key for session management
SECRET_KEY = "your-secret-key"
serializer = URLSafeTimedSerializer(SECRET_KEY)

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse(str(BASE_DIR / "login.html"))

@router.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    with db_service.get_cursor(dictionary=True) as (cursor, conn):
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        # Create a session token
        session_token = serializer.dumps(user['id'])
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session", value=session_token)
        return response
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("session")
    return response

@router.get("/signup", response_class=HTMLResponse)
async def signup_page():
    return FileResponse(str(BASE_DIR / "signup.html"))

@router.post("/signup")
async def signup(name: str = Form(...), email: str = Form(...), phone: str = Form(None), password: str = Form(...)):
    with db_service.get_cursor(dictionary=True) as (cursor, conn):
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (name, email, phone, password) VALUES (%s, %s, %s, %s)", (name, email, phone, hashed_password.decode('utf-8')))
        user_id = cursor.lastrowid
        conn.commit()

    # Log the user in
    session_token = serializer.dumps(user_id)
    response = JSONResponse(content={"message": "Signup successful"})
    response.set_cookie(key="session", value=session_token)
    return response

@router.post("/change-password")
async def change_password(request: Request, old_password: str = Form(...), new_password: str = Form(...)):
    session_token = request.cookies.get("session")
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = serializer.loads(session_token, max_age=3600)  # 1 hour expiration
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

    with db_service.get_cursor(dictionary=True) as (cursor, conn):
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(old_password.encode('utf-8'), user['password'].encode('utf-8')):
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password.decode('utf-8'), user_id))
            conn.commit()
            return {"message": "Password updated successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid old password")
