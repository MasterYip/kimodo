#!/usr/bin/env python3
"""Test a Hugging Face token (reads from stdin, never in command line)."""
import sys, os, httpx

# Read token securely from stdin
token = input("Paste HF token: ").strip()
if not token or not token.startswith("hf_"):
    print("ERROR: Invalid token format (must start with 'hf_')")
    sys.exit(1)

# Test whoami
r = httpx.get(
    "https://hf-mirror.com/api/whoami-v2",
    headers={"Authorization": f"Bearer {token}"},
    timeout=15,
)
print(f"Whoami: HTTP {r.status_code}")

if r.status_code == 200:
    data = r.json()
    print(f"  User:     {data.get('name', '?')}")
    print(f"  Auth:     {data.get('auth', {}).get('type', '?')}")
else:
    print(f"  FAIL: {r.text[:200]}")
    sys.exit(1)

# Test access to Llama 3
r2 = httpx.head(
    "https://hf-mirror.com/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/config.json",
    headers={"Authorization": f"Bearer {token}"},
    timeout=15,
    follow_redirects=True,
)
print(f"\nLlama 3 8B: HTTP {r2.status_code}")
if r2.status_code == 200:
    print("  ✓ Access granted — can use standard text encoder")
elif r2.status_code == 403:
    print("  ✗ Access denied — use TEXT_ENCODER=llm2vec-nf4 instead")
else:
    print(f"  ? Unexpected: {r2.text[:200]}")

# Test NF4 model
r3 = httpx.head(
    "https://hf-mirror.com/Aero-Ex/KIMODO-Meta3_llm2vec_NF4/resolve/main/model.safetensors",
    timeout=15,
    follow_redirects=True,
)
print(f"\nNF4 encoder: HTTP {r3.status_code}")
if r3.status_code == 200:
    print("  ✓ NF4 model accessible (ungated)")
