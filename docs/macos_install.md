# macOS Installation Guide (Apple Silicon)

This guide provides instructions for installing and running Lada on macOS with Apple Silicon (M1/M2/M3/M4).

## Prerequisites

- macOS 11.0 (Big Sur) or later
- Apple Silicon Mac (M1, M2, M3, or M4)
- Command Line Tools for Xcode
- Homebrew (will be installed if not present)

## Installation Steps

### 1. Install Command Line Tools

```bash
xcode-select --install
```

Or update to the latest version:
```bash
softwareupdate --install "Command Line Tools for Xcode"
```

### 2. Install Homebrew

If you don't have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add Homebrew to your PATH:
```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 3. Install System Dependencies

Install GTK4, libadwaita, and other required libraries:

```bash
brew install gtk4 libadwaita pygobject3 pkg-config gobject-introspection \
             gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad \
             gst-plugins-ugly cairo python@3.12
```

### 4. Clone the Repository

```bash
git clone https://github.com/ladaapp/lada.git
cd lada
```

### 5. Create Virtual Environment

Use Homebrew's Python (required for GTK compatibility):

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

### 6. Install Python Dependencies

```bash
pip install uv
uv pip install -e '.[basicvsrpp,gui]'
```

### 7. Download Model Files

Download the required model weights:

```bash
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_accurate.pt'
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_fast.pt'
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.2.0/lada_mosaic_detection_model_v2.pt'
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.6.0/lada_mosaic_restoration_model_generic_v1.2.pth'
```

Or use curl:

```bash
curl -L -o model_weights/lada_mosaic_detection_model_v3.1_accurate.pt 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_accurate.pt'
curl -L -o model_weights/lada_mosaic_detection_model_v3.1_fast.pt 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_fast.pt'
curl -L -o model_weights/lada_mosaic_detection_model_v2.pt 'https://github.com/ladaapp/lada/releases/download/v0.2.0/lada_mosaic_detection_model_v2.pt'
curl -L -o model_weights/lada_mosaic_restoration_model_generic_v1.2.pth 'https://github.com/ladaapp/lada/releases/download/v0.6.0/lada_mosaic_restoration_model_generic_v1.2.pth'
```

### 8. Fix Configuration Directory Permissions (if needed)

If you encounter permission errors:

```bash
sudo chown -R $USER:staff ~/.config
```

## Running Lada

### GUI Application

```bash
source .venv/bin/activate
lada
```

### CLI Application

```bash
source .venv/bin/activate
lada-cli --input video.mp4 --output restored.mp4 --device mps
```

## Apple Silicon GPU Acceleration

Lada automatically detects and uses Apple Silicon GPU (MPS - Metal Performance Shaders) for acceleration. Always use the `--device mps` flag with the CLI for best performance:

```bash
lada-cli --input video.mp4 --device mps
```

## Troubleshooting

### GTK Warnings

You may see GTK warnings about "Broken accounting of active state". These are cosmetic and don't affect functionality.

### Python Version Issues

**Important**: You must use Homebrew's Python, not Anaconda Python. Anaconda Python is not compatible with PyGObject/GTK on macOS.

If you have Anaconda installed, make sure to deactivate it before creating the virtual environment:

```bash
conda deactivate
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
```

### CUDA Errors

If you see CUDA-related errors, these have been fixed in the macOS support branch. The application now properly detects MPS (Apple Silicon GPU) instead of trying to use CUDA.

## Performance Notes

- **MPS (Metal Performance Shaders)**: Provides GPU acceleration on Apple Silicon
- Use `--device mps` for CLI operations to leverage GPU acceleration
- The GUI automatically selects "Apple Silicon GPU" when available

## Convenience Scripts

You can create launcher scripts for easier access:

**launch_gui.sh:**
```bash
#!/bin/bash
cd "$(dirname "$0")"
eval "$(/opt/homebrew/bin/brew shellenv)"
source .venv/bin/activate
lada
```

**launch_cli.sh:**
```bash
#!/bin/bash
cd "$(dirname "$0")"
eval "$(/opt/homebrew/bin/brew shellenv)"
source .venv/bin/activate
lada-cli "$@"
```

Make them executable:
```bash
chmod +x launch_gui.sh launch_cli.sh
```

## Known Issues

1. **PyGObject requires Homebrew Python**: Anaconda Python is not compatible
2. **GTK4 warnings**: Cosmetic warnings from GTK4 on macOS can be ignored
3. **First launch may be slow**: PyGObject and GTK initialization takes time on first run

## Credits

macOS Apple Silicon support contributed by the community. Thanks to all contributors!

## References

- [Lada GitHub Repository](https://github.com/ladaapp/lada)
- [Homebrew Documentation](https://docs.brew.sh/)
- [PyGObject Documentation](https://pygobject.readthedocs.io/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
