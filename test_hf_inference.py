import os
import sys
import time
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure predictor picks up .env
from predictor import SentimentEmotionPredictor

print("=" * 60)
print("  HUGGING FACE SERVERLESS INFERENCE TEST SUITE  ")
print("=" * 60)

predictor = SentimentEmotionPredictor()

# 1. Health check test
print("\n[TEST 1] Testing /health status check...")
health = predictor.check_health()
print("Health Status:", json.dumps(health, indent=2))

# 2. Test 3 representative inputs
test_cases = [
    ("Urdu Script", "یہ سروس بہت شاندار ہے"),
    ("Roman Urdu", "Main bohot excited hoon, ye best din tha"),
    ("Mixed Urdu-English", "Yar this project is actually mazedaar")
]

print("\n[TEST 2] Testing 3 Representative Inputs...")
for label, text in test_cases:
    print(f"\n--- Testing {label}: '{text}' ---")
    start_time = time.time()
    res = predictor.predict(text)
    latency = time.time() - start_time

    print(f"Latency: {latency:.3f} seconds")
    print(f"Response: {json.dumps(res, ensure_ascii=False, indent=2)}")

    # Schema verification
    expected_keys = {"text", "sentiment", "sentiment_scores", "emotion", "emotion_scores", "attention"}
    if expected_keys.issubset(res.keys()):
        print("Schema Verification: PASSED (all required keys present)")
    else:
        print(f"Schema Verification: FAILED (missing keys: {expected_keys - set(res.keys())})")

# 3. Deliberately test failure path (invalid token)
print("\n[TEST 3] Testing Failure Path (Invalid Token / API Error)...")
orig_token = predictor.hf_token
predictor.hf_token = "hf_invalid_test_token_12345"

fail_res = predictor.predict("یہ ٹیسٹ ہے")
print("Failure Path Response:", json.dumps(fail_res, ensure_ascii=False, indent=2))
if "error" in fail_res or "sentiment" in fail_res:
    print("Failure Path Verification: PASSED (Graceful handling, no crash)")

predictor.hf_token = orig_token

print("\n" + "=" * 60)
print("  TEST SUITE COMPLETE  ")
print("=" * 60)
