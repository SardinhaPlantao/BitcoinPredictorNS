import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    import requests
    print(f"requests version: {requests.__version__}")
except ImportError as e:
    print(f"Failed to import requests: {e}")