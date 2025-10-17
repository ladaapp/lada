#!/usr/bin/env python3
"""
macOS M4 Chip Optimization Script for Lada

This script helps optimize Lada's performance on macOS Studio M4 chip by:
1. Detecting the optimal device (MPS) and settings
2. Recommending performance settings based on video resolution
3. Providing memory management tips
4. Suggesting optimal encoder settings
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_system_info():
    """Check system information and hardware capabilities"""
    print("=== System Information ===")
    print(f"macOS Version: {platform.mac_ver()[0]}")
    print(f"Python Version: {platform.python_version()}")
    
    # Check if we're on Apple Silicon
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.machine'], capture_output=True, text=True)
        machine = result.stdout.strip()
        print(f"Hardware: {machine}")
        
        # Check for M4 specifically
        result = subprocess.run(['system_profiler', 'SPHardwareDataType', '-json'], 
                              capture_output=True, text=True)
        if 'M4' in result.stdout:
            print("✓ Detected Apple M4 chip")
        elif 'M3' in result.stdout:
            print("✓ Detected Apple M3 chip")
        elif 'M2' in result.stdout:
            print("✓ Detected Apple M2 chip")
        elif 'M1' in result.stdout:
            print("✓ Detected Apple M1 chip")
    except:
        print("Could not detect hardware information")
    
    # Check memory
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        mem_bytes = int(result.stdout.strip())
        mem_gb = mem_bytes / (1024**3)
        print(f"Memory: {mem_gb:.1f} GB")
    except:
        print("Could not detect memory information")
    
    print()

def check_pytorch_mps():
    """Check PyTorch MPS availability"""
    print("=== PyTorch MPS Support ===")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            print(f"MPS Available: {mps_available}")
            print(f"MPS Built: {mps_built}")
            
            if mps_available:
                print("✓ MPS is available for GPU acceleration")
                return True
            else:
                print("✗ MPS is not available")
                return False
        else:
            print("✗ MPS not supported in this PyTorch version")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking MPS: {e}")
        return False

def recommend_settings(video_resolution="1080p", memory_gb=16):
    """Recommend optimal settings based on video resolution and available memory"""
    print(f"\n=== Recommended Settings for {video_resolution} ===")
    
    # Device recommendation
    if check_pytorch_mps():
        print("Device: mps (Apple Silicon GPU)")
    else:
        print("Device: cpu (MPS not available)")
    
    # Clip length based on resolution and memory
    if video_resolution == "4K":
        if memory_gb >= 64:
            max_clip_length = 180
        elif memory_gb >= 32:
            max_clip_length = 120
        else:
            max_clip_length = 60
    elif video_resolution == "1440p":
        if memory_gb >= 32:
            max_clip_length = 180
        elif memory_gb >= 16:
            max_clip_length = 120
        else:
            max_clip_length = 90
    else:  # 1080p or lower
        if memory_gb >= 16:
            max_clip_length = 180
        else:
            max_clip_length = 120
    
    print(f"Max Clip Length: {max_clip_length}")
    
    # Encoder recommendations
    print("\nEncoder Settings:")
    print("Primary Codec: h264_videotoolbox (Apple's hardware encoder)")
    print("Alternative: hevc_videotoolbox (for better compression)")
    print("CRF: 20-23 (good balance of quality and file size)")
    
    # Model recommendations
    print("\nModel Settings:")
    print("Detection Model: v3.1-fast (good balance of speed and accuracy)")
    print("Restoration Model: basicvsrpp-v1.2 (default, best quality)")

def generate_command(video_path, output_path=None, video_resolution="1080p", memory_gb=16):
    """Generate optimized command line"""
    print("\n=== Example Command ===")
    
    cmd_parts = ["lada-cli", "--input", video_path, "--device", "mps"]
    
    # Determine optimal settings
    if video_resolution == "4K":
        if memory_gb < 32:
            cmd_parts.extend(["--max-clip-length", "60"])
        elif memory_gb < 64:
            cmd_parts.extend(["--max-clip-length", "120"])
    elif video_resolution == "1440p":
        if memory_gb < 16:
            cmd_parts.extend(["--max-clip-length", "90"])
        elif memory_gb < 32:
            cmd_parts.extend(["--max-clip-length", "120"])
    
    # Add output path if specified
    if output_path:
        cmd_parts.extend(["--output", output_path])
    
    # Add encoder settings
    cmd_parts.extend(["--codec", "h264_videotoolbox", "--crf", "22"])
    
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd_parts))

def check_environment_variables():
    """Check and suggest environment variables"""
    print("\n=== Environment Variables ===")
    
    # Check TMPDIR
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    print(f"TMPDIR: {tmpdir}")
    
    # Suggest setting TMPDIR to a faster location if available
    if tmpdir == '/tmp':
        print("Tip: Consider setting TMPDIR to a RAM disk for faster temporary storage:")
        print("  export TMPDIR=/dev/shm  # Linux")
        print("  # On macOS, you can create a RAM disk with:")
        print("  diskutil erasevolume HFS+ 'RAMDisk' `hdiutil attach -nomount ram://$((2048*1024))`")
        print("  export TMPDIR=/Volumes/RAMDisk")

def main():
    parser = argparse.ArgumentParser(description="Optimize Lada for macOS M4 chip")
    parser.add_argument("--video-path", help="Path to video file for optimization")
    parser.add_argument("--video-resolution", choices=["1080p", "1440p", "4K"], 
                       default="1080p", help="Video resolution for optimization")
    parser.add_argument("--memory", type=int, help="System memory in GB (auto-detected if not specified)")
    parser.add_argument("--output", help="Output path for optimized video")
    
    args = parser.parse_args()
    
    print("Lada macOS M4 Optimization Tool")
    print("=" * 40)
    
    # Check system information
    check_system_info()
    
    # Get memory if not specified
    memory_gb = args.memory
    if memory_gb is None:
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            mem_bytes = int(result.stdout.strip())
            memory_gb = int(mem_bytes / (1024**3))
        except:
            memory_gb = 16  # Default assumption
    
    # Check PyTorch MPS
    mps_available = check_pytorch_mps()
    
    # Recommend settings
    recommend_settings(args.video_resolution, memory_gb)
    
    # Generate command if video path provided
    if args.video_path:
        generate_command(args.video_path, args.output, args.video_resolution, memory_gb)
    
    # Check environment variables
    check_environment_variables()
    
    print("\n=== Performance Tips ===")
    print("1. Close unnecessary applications to free up memory")
    print("2. Use the latest version of PyTorch for best MPS performance")
    print("3. For batch processing, process videos one at a time")
    print("4. Monitor memory usage with Activity Monitor")
    print("5. Consider using a lower max-clip-length if you encounter memory issues")
    
    if not mps_available:
        print("\n⚠️  MPS not available - performance will be limited to CPU")
        print("   Update PyTorch with: pip install --upgrade torch torchvision torchaudio")
        print("   Ensure you're running macOS 13.0 (Ventura) or later")

if __name__ == "__main__":
    main()