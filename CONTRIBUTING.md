# Contributing to lada

Thank you for your interest in contributing to lada!

## Development Setup

1. **Requirements**
   - Python 3.10+
   - Git

2. **Clone the repository**
   ```bash
   git clone https://github.com/ladaapp/lada.git
   cd lada
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # if available
   ```

5. **Run tests**
   ```bash
   pytest tests/
   ```

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Code style**
   - Follow PEP 8
   - Use type hints
   - Add docstrings

3. **Commit and push**
   ```bash
   git commit -m "feat: add your feature"
   git push origin feat/your-feature-name
   ```

## Pull Request Process

1. Fork the repository
2. Create your feature branch
3. Make your changes with tests
4. Submit a PR with description

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include Python version and OS
- Provide reproduction steps

## License

By contributing, you agree that your contributions will be licensed under the project license.
