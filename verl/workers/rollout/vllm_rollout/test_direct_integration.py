#!/usr/bin/env python3
"""
Test script for direct ArcticInference integration

This script tests the problem_id functionality directly integrated 
into ArcticInference model_runner.py
"""

import sys
import os

def test_context_manager_import():
    """Test if ProblemIdContextManager can be imported."""
    try:
        # Test the same import path that rollout uses
        from rllm.ArcticInference.arctic_inference.vllm.model_runner import ProblemIdContextManager
        print("✓ ProblemIdContextManager import successful")
        return True, ProblemIdContextManager
    except ImportError as e:
        print(f"✗ ProblemIdContextManager import failed: {e}")
        print("This is expected if ArcticInference is not properly installed")
        return False, None

def test_context_functionality(ProblemIdContextManager):
    """Test the context manager functionality."""
    try:
        # Test data
        problem_ids = ["math_001", "code_002", None, "physics_003"]
        
        # Test context manager
        with ProblemIdContextManager.batch_context(problem_ids):
            # Test retrieval
            assert ProblemIdContextManager.get_problem_id_for_index(0) == "math_001"
            assert ProblemIdContextManager.get_problem_id_for_index(1) == "code_002"
            assert ProblemIdContextManager.get_problem_id_for_index(2) is None
            assert ProblemIdContextManager.get_problem_id_for_index(3) == "physics_003"
            assert ProblemIdContextManager.get_problem_id_for_index(4) is None  # Out of range
        
        # After context, should be cleared
        assert ProblemIdContextManager.get_problem_id_for_index(0) is None
        
        print("✓ Context manager functionality test successful")
        return True
    except Exception as e:
        print(f"✗ Context manager functionality test failed: {e}")
        return False

def test_rollout_integration():
    """Test if rollout can import the context manager."""
    try:
        # Test the import that rollout uses
        from rllm.ArcticInference.arctic_inference.vllm.model_runner import ProblemIdContextManager
        
        # Simulate what rollout does
        problem_ids = ["test_001", "test_002"]
        
        with ProblemIdContextManager.batch_context(problem_ids):
            # Simulate model_runner getting problem_id
            problem_id = ProblemIdContextManager.get_problem_id_for_index(0)
            assert problem_id == "test_001"
        
        print("✓ Rollout integration test successful")
        return True
    except Exception as e:
        print(f"✗ Rollout integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Direct ArcticInference Integration")
    print("=" * 50)
    
    # Test 1: Import
    success, ProblemIdContextManager = test_context_manager_import()
    if not success:
        print("Cannot proceed without successful import")
        return False
    
    # Test 2: Functionality
    success = test_context_functionality(ProblemIdContextManager)
    if not success:
        return False
    
    # Test 3: Integration
    success = test_rollout_integration()
    if not success:
        return False
    
    print("\n🎉 All tests passed! Direct integration is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)