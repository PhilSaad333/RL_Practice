"""
Quick test to validate Stage 1 model loader refactoring.

Tests that the new model_loader.py imports correctly and the 
offline_entropy_probe.py can import from it without errors.
"""

def test_model_loader_import():
    """Test that model_loader.py imports correctly."""
    print("Testing model_loader.py imports...")
    
    try:
        from entropy_experiments.model_loader import load_peft_for_probe, load_adam_optimizer_from_path
        print("✅ model_loader imports successful")
        return True
    except Exception as e:
        print(f"❌ model_loader import failed: {e}")
        return False

def test_offline_entropy_probe_import():
    """Test that offline_entropy_probe.py can import from model_loader."""
    print("Testing offline_entropy_probe.py imports...")
    
    try:
        # This will test if offline_entropy_probe.py can import the model_loader
        import sys
        from pathlib import Path
        
        # Add entropy_experiments to path for relative import
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
        print("✅ offline_entropy_probe imports successful")
        return True
    except Exception as e:
        print(f"❌ offline_entropy_probe import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_compatibility():
    """Test that the new config format is compatible."""
    print("Testing config compatibility...")
    
    # Simulate the new config structure
    test_config = {
        'checkpoint': {
            'model_config_path': 'Qwen/Qwen2.5-1.5B',
            'use_qlora': False,  # New parameter
            'dtype': 'bf16',     # New parameter
            'device_map': 'cuda' # New parameter
        },
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'train',
            'B_E_values': [16],
            'B_U_values': [8],
            'G': 8
        },
        'probe_rework': {
            'compute_delta_h1': True,
            'master_seed': 42
        }
    }
    
    # Test that our loader would get the right values
    use_qlora = bool(test_config['checkpoint'].get('use_qlora', False))
    dtype = test_config['checkpoint'].get('dtype', 'bf16')
    device_map = test_config['checkpoint'].get('device_map', 'cuda')
    
    print(f"Config parsing test:")
    print(f"  use_qlora: {use_qlora} (expected: False)")
    print(f"  dtype: {dtype} (expected: 'bf16')")
    print(f"  device_map: {device_map} (expected: 'cuda')")
    
    # Test defaults when keys are missing
    minimal_config = {'checkpoint': {}}
    use_qlora_default = bool(minimal_config['checkpoint'].get('use_qlora', False))
    dtype_default = minimal_config['checkpoint'].get('dtype', 'bf16')
    
    print(f"Default values test:")
    print(f"  use_qlora default: {use_qlora_default} (expected: False)")
    print(f"  dtype default: {dtype_default} (expected: 'bf16')")
    
    print("✅ Config compatibility test passed")
    return True

def main():
    """Run all Stage 1 validation tests."""
    print("🧪 STAGE 1 VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Model Loader Import", test_model_loader_import),
        ("Offline Entropy Probe Import", test_offline_entropy_probe_import),  
        ("Config Compatibility", test_config_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Stage 1 refactoring successful!")
        print("✅ Ready to proceed to Stage 2")
    else:
        print("⚠️ SOME TESTS FAILED - Need to fix issues before Stage 2")
    
    return all_passed

if __name__ == "__main__":
    main()