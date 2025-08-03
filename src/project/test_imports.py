"""
Simple test script to isolate import issues.
"""

import sys
import os
from pathlib import Path

# Add project directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🔍 Testing imports step by step...")

try:
    print("1. Testing debug_utils...")
    from debug_utils import create_logger, SystemMonitor
    print("   ✅ debug_utils imported successfully")
except Exception as e:
    print(f"   ❌ debug_utils failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing training_utils...")
    from training_utils import DatasetLoader
    print("   ✅ training_utils imported successfully")
except Exception as e:
    print(f"   ❌ training_utils failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing ParametricOptimizer...")
    from parametric_optimization import ParametricOptimizer
    print("   ✅ ParametricOptimizer imported successfully")
except Exception as e:
    print(f"   ❌ ParametricOptimizer failed: {e}")
    import traceback
    traceback.print_exc()

print("🎯 Import test completed!")
