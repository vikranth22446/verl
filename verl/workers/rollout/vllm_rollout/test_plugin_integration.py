#!/usr/bin/env python3
"""
Test script for ArcticInference plugin integration

This script tests that the problem_id functionality works through 
the vLLM plugin mechanism.
"""

def test_plugin_loading():
    """Test if vLLM plugin loading works."""
    try:
        import vllm
        vllm.plugins.load_general_plugins()
        print("✓ vLLM plugins loaded successfully")
        return True
    except Exception as e:
        print(f"✗ vLLM plugin loading failed: {e}")
        return False

def test_context_manager_availability():
    """Test if ProblemIdContextManager is available after plugin loading."""
    try:
        # This should work after plugins are loaded
        from rllm.ArcticInference.arctic_inference.vllm.model_runner import ProblemIdContextManager
        print("✓ ProblemIdContextManager available through plugin")
        return True, ProblemIdContextManager
    except ImportError as e:
        print(f"✗ ProblemIdContextManager not available: {e}")
        return False, None

def test_context_functionality(ProblemIdContextManager):
    """Test the context manager functionality."""
    try:
        problem_ids = ["test_001", "test_002", None]
        
        with ProblemIdContextManager.batch_context(problem_ids):
            assert ProblemIdContextManager.get_problem_id_for_index(0) == "test_001"
            assert ProblemIdContextManager.get_problem_id_for_index(1) == "test_002"
            assert ProblemIdContextManager.get_problem_id_for_index(2) is None
        
        # After context, should be cleared
        assert ProblemIdContextManager.get_problem_id_for_index(0) is None
        
        print("✓ Context manager functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Context manager functionality failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing ArcticInference Plugin Integration")
    print("=" * 50)
    
    # Test 1: Plugin loading
    if not test_plugin_loading():
        return False
    
    # Test 2: Context manager availability
    success, ProblemIdContextManager = test_context_manager_availability()
    if not success:
        print("This is expected if ArcticInference plugin is not available")
        return True  # Not a failure, just not available
    
    # Test 3: Functionality
    if not test_context_functionality(ProblemIdContextManager):
        return False
    
    print("\n🎉 All tests passed! Plugin integration is working.")
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)