import requests
import json
import time

API_URL = "http://localhost:8000"

def wait_for_api():
    """Waits for the API to be available."""
    print("Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"API is ready! Models loaded: {data.get('models_loaded', [])}")
                if not data.get('models_loaded'):
                    print("Warning: No models loaded yet. Predictions may fail.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("API failed to start in time or is unreachable.")
    return False

def test_prediction(text, data_type):
    """Tests the prediction endpoint."""
    print(f"\nTesting {data_type} with input: '{text}'")
    try:
        payload = {"text": text, "data_type": data_type}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    if wait_for_api():
        # Test SMS
        test_prediction("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!", "sms")
        test_prediction("Hey, are we still executed for dinner tonight?", "sms")

        # Test Email
        test_prediction("Dear user, your account has been compromised. Click here to reset password.", "email")
        test_prediction("Meeting minutes attached. Best regards.", "email")
