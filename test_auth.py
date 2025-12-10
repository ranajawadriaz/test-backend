"""
Test script for authentication system
Run this to verify registration, login, and protected endpoints
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_authentication():
    print("=" * 60)
    print("Testing Audio Deepfake Detection Authentication System")
    print("=" * 60)
    
    # Test data
    test_user = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123",
        "full_name": "Test User"
    }
    
    # 1. Test Registration
    print("\n1️⃣  Testing User Registration...")
    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=test_user)
        if response.status_code == 201:
            print("✅ Registration successful!")
            data = response.json()
            token = data["access_token"]
            print(f"   Token: {token[:50]}...")
            print(f"   Expires in: {data['expires_in']} seconds (2 hours)")
            print(f"   User: {data['user']['username']} ({data['user']['email']})")
        elif response.status_code == 400:
            print("⚠️  User already exists, trying login instead...")
            # If user exists, try login
            login_data = {
                "username": test_user["username"],
                "password": test_user["password"]
            }
            response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
            if response.status_code == 200:
                print("✅ Login successful!")
                data = response.json()
                token = data["access_token"]
                print(f"   Token: {token[:50]}...")
            else:
                print(f"❌ Login failed: {response.text}")
                return
        else:
            print(f"❌ Registration failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Test Token Validation
    print("\n2️⃣  Testing Token Validation...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/auth/validate", headers=headers)
        if response.status_code == 200:
            print("✅ Token is valid!")
            print(f"   {response.json()}")
        else:
            print(f"❌ Token validation failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Test Get Current User
    print("\n3️⃣  Testing Get Current User Info...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        if response.status_code == 200:
            print("✅ User info retrieved!")
            user = response.json()
            print(f"   Username: {user['username']}")
            print(f"   Email: {user['email']}")
            print(f"   Created: {user['created_at']}")
        else:
            print(f"❌ Failed to get user info: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Test Protected Endpoint (without audio file)
    print("\n4️⃣  Testing Protected Prediction Endpoint...")
    print("   ℹ️  Note: This will fail without an audio file, but tests authentication")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Try to access predict endpoint (will fail due to missing file, but tests auth)
        response = requests.post(f"{BASE_URL}/predict", headers=headers)
        if response.status_code == 422:
            print("✅ Authentication passed! (Failed on missing file as expected)")
        elif response.status_code == 401:
            print("❌ Authentication failed!")
        else:
            print(f"   Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 5. Test Unauthorized Access
    print("\n5️⃣  Testing Unauthorized Access (without token)...")
    try:
        # Try to access protected endpoint without token
        response = requests.post(f"{BASE_URL}/predict")
        if response.status_code == 403:
            print("✅ Correctly blocked unauthorized access!")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 6. Test Logout
    print("\n6️⃣  Testing Logout...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BASE_URL}/auth/logout", headers=headers)
        if response.status_code == 200:
            print("✅ Logout successful!")
            print(f"   {response.json()['message']}")
        else:
            print(f"❌ Logout failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Authentication system test completed!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   - Token expires after 2 hours")
    print("   - Use the token in Authorization header: Bearer <token>")
    print("   - Visit http://localhost:8000/docs for interactive API docs")
    print("=" * 60)

if __name__ == "__main__":
    print("\n⚠️  Make sure the backend server is running!")
    print("   Start it with: uvicorn main:app --reload\n")
    
    input("Press Enter to start testing...")
    test_authentication()
