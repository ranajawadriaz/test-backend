"""
Pydantic Schemas for Request/Response Validation
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
import re

class UserRegister(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match("^[a-zA-Z0-9_-]+$", v):
            raise ValueError('Username must contain only letters, numbers, underscores, and hyphens')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    """Schema for user response (without password)"""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for token response"""
    access_token: str
    token_type: str
    expires_in: int  # Seconds until expiration
    user: UserResponse

class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    email: Optional[str] = None
