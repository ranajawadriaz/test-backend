"""
Database Models for User Authentication and Prediction History
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
from database import Base
import datetime

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    """
    User model for authentication
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def verify_password(self, password: str) -> bool:
        """
        Verify password against hashed password
        """
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Hash a password for storing
        """
        return pwd_context.hash(password)
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"


class Prediction(Base):
    """
    Prediction model to store user prediction history
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer)
    duration_seconds = Column(Float)
    
    # Final prediction results
    final_prediction = Column(String(20), nullable=False)  # BONAFIDE or SPOOF
    final_confidence = Column(Float, nullable=False)
    confidence_level = Column(String(20))  # HIGH, MEDIUM, LOW
    models_agree = Column(Boolean)
    
    # Individual model predictions (stored as JSON-like strings)
    rf_prediction = Column(String(20))
    rf_confidence = Column(Float)
    cnn_prediction = Column(String(20))
    cnn_confidence = Column(Float)
    ensemble_prediction = Column(String(20))
    ensemble_confidence = Column(Float)
    
    # Processing info
    num_chunks = Column(Integer)
    processing_time = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to User
    user = relationship("User", backref="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, user_id={self.user_id}, prediction={self.final_prediction})>"
