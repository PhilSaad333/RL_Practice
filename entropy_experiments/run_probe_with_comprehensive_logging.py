#!/usr/bin/env python3
"""
Comprehensive logging wrapper for entropy probe to identify crash location.
"""

import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
import yaml

def setup_comprehensive_logging():
    """Set up comprehensive logging to capture everything."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("entropy_experiments/logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"entropy_probe_comprehensive_{timestamp}.log"
    
    # Set up root logger to capture everything
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('entropy_probe_comprehensive')
    logger.info(f"📝 Comprehensive logging started - Log file: {log_file}")
    return logger, log_file

def run_entropy_probe_with_logging():
    """Run the full entropy probe with comprehensive logging."""
    logger, log_file = setup_comprehensive_logging()
    
    try:
        logger.info("🚀 STARTING COMPREHENSIVE ENTROPY PROBE RUN")
        logger.info("=" * 70)
        
        # Step 1: Import and basic setup
        logger.info("📦 Step 1: Importing modules...")
        from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
        logger.info("✅ Successfully imported OfflineEntropyProbe")
        
        # Step 2: Load config
        logger.info("📋 Step 2: Loading configuration...")
        config_path = "entropy_experiments/configs/test_deltaH1.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config loaded from {config_path}")
        logger.info(f"   B_E={config['batch_config']['B_E_values']}")
        logger.info(f"   B_U={config['batch_config']['B_U']}")
        logger.info(f"   G={config['batch_config']['G']}")
        logger.info(f"   Expected sequences: E={config['batch_config']['B_E_values'][0]}×1 + U={config['batch_config']['B_U']}×{config['batch_config']['G']} = {config['batch_config']['B_E_values'][0] + config['batch_config']['B_U']*config['batch_config']['G']}")
        
        # Step 3: Create probe instance
        logger.info("🏗️ Step 3: Creating OfflineEntropyProbe instance...")
        probe = OfflineEntropyProbe(config)
        logger.info("✅ Probe instance created successfully")
        
        # Step 4: Run the full mixed probe analysis
        logger.info("🎯 Step 4: Starting FULL mixed probe analysis (this will take time)...")
        logger.info("   This includes: model loading, optimizer loading, E/U sampling, gradient computation, deltaH1 calculation")
        
        start_time = time.time()
        
        # Call the main method that does everything
        result = probe.run_mixed_probe()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("🎉 SUCCESS! Entropy probe completed successfully!")
        logger.info(f"⏱️ Total runtime: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info("📊 RESULTS:")
        if isinstance(result, dict):
            for key, value in result.items():
                logger.info(f"   {key}: {value}")
        else:
            logger.info(f"   Result: {result}")
        
        logger.info("=" * 70)
        logger.info(f"✅ Complete log saved to: {log_file}")
        
        return True, result
        
    except Exception as e:
        logger.error("💥 ENTROPY PROBE CRASHED!")
        logger.error(f"Error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Also print crash info to make it super visible
        print("\n" + "="*70)
        print("💥 CRASH DETECTED!")
        print(f"Error: {str(e)}")
        print(f"See full details in log: {log_file}")
        print("="*70)
        
        return False, None

if __name__ == "__main__":
    print("🔍 Starting comprehensive entropy probe run with full logging...")
    success, result = run_entropy_probe_with_logging()
    
    if success:
        print("🎉 Entropy probe completed successfully!")
        sys.exit(0)
    else:
        print("💥 Entropy probe crashed - check logs for details")
        sys.exit(1)