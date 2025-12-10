"""
Authentication Routes and Dependencies
Handles user registration, login, logout, and token validation
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta
import logging

from database import get_db
from models import User
from schemas import UserRegister, UserLogin, Token, UserResponse, TokenData
from jwt_ import create_access_token, verify_token, ACCESS_TOKEN_EXPIRE_HOURS

logger = logging.getLogger(__name__)

# Create router for auth endpoints
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme for JWT Bearer tokens
security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token
    Automatically checks token expiry (2 hours)
    """
    token = credentials.credentials
    
    # Verify and decode token (raises HTTPException if expired or invalid)
    payload = verify_token(token)
    
    # Extract user info from token
    user_id: int = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    
    return user

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user account
    
    - **username**: Unique username (3-50 chars, alphanumeric)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, 1 upper, 1 lower, 1 digit)
    - **full_name**: Optional full name
    """
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = User.get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.username}")
    
    # Create access token (2-hour expiry)
    access_token = create_access_token(
        data={
            "user_id": new_user.id,
            "username": new_user.username,
            "email": new_user.email
        }
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600,  # Convert to seconds
        "user": new_user
    }

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Login with username and password
    Returns JWT token valid for 2 hours
    
    - **username**: Your username
    - **password**: Your password
    """
    # Find user by username
    user = db.query(User).filter(User.username == user_data.username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not user.verify_password(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    
    logger.info(f"User logged in: {user.username}")
    
    # Create access token (2-hour expiry)
    access_token = create_access_token(
        data={
            "user_id": user.id,
            "username": user.username,
            "email": user.email
        }
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600,  # Convert to seconds
        "user": user
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user
    Note: Client should discard the JWT token
    Token will automatically expire after 2 hours
    """
    logger.info(f"User logged out: {current_user.username}")
    
    return {
        "message": "Successfully logged out",
        "detail": "Please discard your authentication token"
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information
    Requires valid JWT token (Bearer authentication)
    """
    return current_user

@router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_user)):
    """
    Validate if current token is still valid
    Returns user info if token is valid, 401 if expired
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "message": "Token is valid"
    }
