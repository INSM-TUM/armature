# Armature

A workbench for process scientists — a unified Python ecosystem for Activity Relationship Matrix (ARM) analysis, discovery, and process redesign.

## Core Value Proposition

**Transparent process analysis** — users can see exactly why the system reached every conclusion it does, down to the raw counts, scores, intermediate structures, and rule traces.

Armature consolidates logic previously fragmented across Rust, JavaScript, and Python repositories into a single, maintainable, and test-driven codebase. It prioritizes transparency over abstraction, making it ideal for researchers who need to understand and tune process mining algorithms.

## Features

- **Discovery Algorithm**: Convert XES event logs to Activity Relationship Matrices with SCC detection and NAND/OR pattern recognition
- **Classification Engine**: Analyze process structure with transparent rule traces and ratio debugging
- **CLI Interface**: Command-line tools for automation (`discover`, `classify`, `inspect`, `weights`)
- **Web Dashboard**: Interactive visualization with matrix grid view and inspector panel
- **Matrix Operations**: Compare and diff matrices to highlight process changes

## Requirements

- **Python 3.10+**
- Virtual environment (recommended for all installations)

## Installation

**Important:** Use a virtual environment to avoid conflicts with system-managed Python installations.

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install package

```bash
# Standard installation
pip install -e .

# Development installation (includes pytest and testing tools)
pip install -e ".[dev]"
```

### 3. Verify installation

```bash
# Check package imports
python -c "import armature; print(f'Armature v{armature.__version__}')"

# Run test suite
pytest -v
```

## Quick Start

```bash
# Discover ARM matrix from event log
armature discover log.xes

# Show dependency weights for all activity pairs
armature weights log.xes

# Classify a process matrix
armature classify matrix.yaml

# Inspect matrix details
armature inspect matrix.yaml
```

## Commands

| Command    | Input        | Description                                              |
|------------|--------------|----------------------------------------------------------|
| `discover` | XES log      | Extract ARM matrix; output YAML or JSON                  |
| `weights`  | XES log      | Compute W = N_D/T for every dependency type per pair     |
| `classify` | YAML matrix  | Classify process structure with rule traces              |
| `inspect`  | YAML matrix  | Full transparency dump: counts, ratios, dependency cells |

## License

MIT
