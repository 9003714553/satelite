
import sys
import os
try:
    from models import UNetGenerator
    print("SUCCESS: Imported UNetGenerator")
except ImportError as e:
    print(f"FAILURE: Could not import UNetGenerator. Error: {e}")
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")
