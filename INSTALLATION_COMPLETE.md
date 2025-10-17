# Lada Installation Guide - macOS Setup Complete! 🎉

## ✅ What Has Been Installed

### System Dependencies
- ✅ **Homebrew** - macOS package manager
- ✅ **Command Line Tools** - Updated to version 16.4
- ✅ **GTK4 & libadwaita** - GUI framework
- ✅ **GStreamer** - Media framework
- ✅ **PyGObject** - Python GTK bindings
- ✅ **Python 3.12** (Homebrew version for GTK compatibility)

### Lada Application
- ✅ **Lada CLI** - Command-line interface
- ✅ **Lada GUI** - Graphical user interface
- ✅ **BasicVSR++ Support** - Advanced restoration model
- ✅ **Model Files Downloaded**:
  - `lada_mosaic_detection_model_v3.1_accurate.pt` (20MB)
  - `lada_mosaic_detection_model_v3.1_fast.pt` (5.7MB)
  - `lada_mosaic_detection_model_v2.pt` (43MB)
  - `lada_mosaic_restoration_model_generic_v1.2.pth` (75MB)

## 🚀 How to Run Lada

### Option 1: Launch GUI (Easy Way)
```bash
./launch_gui.sh
```

Or navigate to the directory and run:
```bash
cd "/Users/ry/Downloads/lada-main mac"
source .venv-gui/bin/activate
lada
```

### Option 2: Use CLI (Command Line)
```bash
./launch_cli.sh --input your_video.mp4 --output restored.mp4 --device mps
```

Or for more control:
```bash
cd "/Users/ry/Downloads/lada-main mac"
source .venv-gui/bin/activate
lada-cli --input your_video.mp4 --output restored.mp4 --device mps
```

## 📋 CLI Usage Examples

### Basic restoration:
```bash
./launch_cli.sh --input video.mp4
```

### With Apple Silicon GPU acceleration:
```bash
./launch_cli.sh --input video.mp4 --device mps
```

### Custom output location:
```bash
./launch_cli.sh --input video.mp4 --output restored.mp4 --device mps
```

### Process entire directory:
```bash
./launch_cli.sh --input /path/to/videos/ --output /path/to/output/ --device mps
```

### List available devices:
```bash
./launch_cli.sh --list-devices
```

### List available models:
```bash
./launch_cli.sh --list-mosaic-detection-models
./launch_cli.sh --list-mosaic-restoration-models
```

## 🔧 Important Notes

### Python Environment
- **GUI/CLI use Homebrew Python** (`.venv-gui/`) - Required for GTK compatibility
- **Anaconda Python** is not compatible with PyGObject/GTK on macOS
- Always activate the correct environment before running commands

### Apple Silicon (M1/M2/M3/M4) Optimization
- Use `--device mps` to leverage Metal Performance Shaders
- This significantly accelerates processing on Apple Silicon Macs

### Model Files Location
All model files are in: `/Users/ry/Downloads/lada-main mac/model_weights/`

## 🐛 Troubleshooting

### GUI doesn't launch:
```bash
# Make sure you're using the correct environment
cd "/Users/ry/Downloads/lada-main mac"
source .venv-gui/bin/activate
lada
```

### Command not found:
```bash
# Make launcher scripts executable
chmod +x launch_gui.sh launch_cli.sh
```

### Import errors:
```bash
# Reinstall dependencies
cd "/Users/ry/Downloads/lada-main mac"
source .venv-gui/bin/activate
uv pip install -e '.[gui,basicvsrpp]'
```

## 📁 Directory Structure

```
lada-main mac/
├── launch_gui.sh          # GUI launcher script
├── launch_cli.sh          # CLI launcher script  
├── .venv-gui/             # Python virtual environment (Homebrew Python)
├── model_weights/         # AI model files
├── lada/                  # Source code
├── pyproject.toml         # Project configuration
└── README.md             # Original documentation
```

## 🎯 Next Steps

1. **Test the GUI**: Run `./launch_gui.sh` to explore the interface
2. **Test the CLI**: Process a sample video with `./launch_cli.sh --input test.mp4 --device mps`
3. **Read the docs**: Check `docs/` for advanced features and training

## 📚 Additional Resources

- GitHub: https://github.com/ladaapp/lada
- Documentation: See `docs/` folder
- Issues: Report bugs on GitHub

---

**Installation completed on:** October 17, 2025  
**Python version:** 3.12.12 (Homebrew)  
**Platform:** macOS (Apple Silicon)
