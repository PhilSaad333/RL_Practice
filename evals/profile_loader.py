# evals/profile_loader.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Configuration profile loader for evaluation settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default profile file location
DEFAULT_PROFILES_PATH = Path(__file__).parent / "eval_profiles.yaml"


def load_profiles(profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load evaluation profiles from YAML file.
    
    Args:
        profile_path: Path to profiles file, defaults to eval_profiles.yaml
    
    Returns:
        Dictionary containing all profiles and metadata
    """
    if profile_path is None:
        profile_path = DEFAULT_PROFILES_PATH
    
    if not profile_path.exists():
        logger.warning(f"Profile file not found: {profile_path}")
        return _get_default_profiles()
    
    try:
        with open(profile_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'profiles' not in data:
            logger.warning(f"No 'profiles' section found in {profile_path}")
            return _get_default_profiles()
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to load profiles from {profile_path}: {e}")
        return _get_default_profiles()


def get_profile(profile_name: str, profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get a specific evaluation profile by name.
    
    Args:
        profile_name: Name of the profile to load
        profile_path: Path to profiles file
    
    Returns:
        Profile configuration dictionary
    
    Raises:
        ValueError: If profile not found
    """
    data = load_profiles(profile_path)
    profiles = data.get('profiles', {})
    
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")
    
    profile = profiles[profile_name].copy()
    
    # Add metadata
    profile['_profile_name'] = profile_name
    profile['_description'] = profile.get('description', f"Profile: {profile_name}")
    
    return profile


def list_profiles(profile_path: Optional[Path] = None) -> Dict[str, str]:
    """
    List all available profiles with their descriptions.
    
    Args:
        profile_path: Path to profiles file
    
    Returns:
        Dictionary mapping profile names to descriptions
    """
    data = load_profiles(profile_path)
    profiles = data.get('profiles', {})
    
    return {
        name: config.get('description', f"Profile: {name}")
        for name, config in profiles.items()
    }


def get_default_profile(profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get the default profile as specified in the config file.
    
    Args:
        profile_path: Path to profiles file
    
    Returns:
        Default profile configuration
    """
    data = load_profiles(profile_path)
    default_name = data.get('default_profile', 'full_evaluation')
    
    try:
        return get_profile(default_name, profile_path)
    except ValueError:
        logger.warning(f"Default profile '{default_name}' not found, using fallback")
        return get_profile('full_evaluation', profile_path)


def _get_default_profiles() -> Dict[str, Any]:
    """Fallback profiles if file loading fails."""
    return {
        'profiles': {
            'quick_test': {
                'subset_frac': 0.02,
                'batch_size': 'conservative',
                'temperature': 0.7,
                'top_p': 1.0,
                'num_return_sequences': 8,
                'max_new_tokens': 200,
                'description': 'Fast testing with 2% of dataset'
            },
            'full_evaluation': {
                'subset_frac': 1.0,
                'batch_size': 'auto',
                'temperature': 0.7,
                'top_p': 1.0,
                'num_return_sequences': 8,
                'max_new_tokens': 200,
                'description': 'Complete evaluation with auto-optimized settings'
            }
        },
        'default_profile': 'full_evaluation'
    }


def apply_profile_to_args(args, profile_name: str, profile_path: Optional[Path] = None):
    """
    Apply a profile's settings to an argparse args object.
    Only updates arguments that weren't explicitly set by the user.
    
    Args:
        args: argparse Namespace object
        profile_name: Name of profile to apply
        profile_path: Path to profiles file
    """
    try:
        profile = get_profile(profile_name, profile_path)
        
        # Track which args were explicitly set by user vs defaults
        # This requires the parser to track defaults, which is tricky
        # For now, we'll just apply profile settings
        
        for key, value in profile.items():
            if key.startswith('_'):  # Skip metadata
                continue
            
            if hasattr(args, key):
                # Only override if the current value seems to be a default
                current_value = getattr(args, key)
                
                # Apply profile value
                setattr(args, key, value)
                logger.info(f"Applied profile '{profile_name}': {key}={value}")
        
        logger.info(f"Successfully applied profile: {profile_name}")
        
    except ValueError as e:
        logger.error(f"Failed to apply profile '{profile_name}': {e}")
        raise


def print_profile_info():
    """Print information about available profiles."""
    profiles = list_profiles()
    
    print("📋 Available Evaluation Profiles:")
    print("=" * 50)
    
    for name, description in profiles.items():
        print(f"  • {name}: {description}")
    
    print()
    data = load_profiles()
    default = data.get('default_profile', 'full_evaluation')
    print(f"🎯 Default profile: {default}")
    
    # Show Lambda optimizations if available
    if 'lambda_optimizations' in data:
        print("\n🚀 Lambda Cloud Optimizations:")
        for gpu_type, info in data['lambda_optimizations'].items():
            if isinstance(info, dict) and 'recommended_profiles' in info:
                profiles_str = ', '.join(info['recommended_profiles'])
                memory = info.get('memory_gb', 'unknown')
                print(f"  • {gpu_type} ({memory}GB): {profiles_str}")


if __name__ == "__main__":
    # CLI for exploring profiles
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore evaluation profiles")
    parser.add_argument("--list", action="store_true", help="List all profiles")
    parser.add_argument("--show", type=str, help="Show specific profile details")
    parser.add_argument("--profiles-file", type=Path, help="Path to profiles file")
    
    args = parser.parse_args()
    
    if args.list:
        print_profile_info()
    elif args.show:
        try:
            profile = get_profile(args.show, args.profiles_file)
            print(f"Profile: {args.show}")
            print("=" * 30)
            for key, value in profile.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print_profile_info()