"""
Survey Archetypes - New Project Structure

This script creates the complete folder structure for the refactored project.
Run this first to set up all directories.
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define the structure
    structure = {
        'data': {
            'input': {},
            'personas': {},
            'output': {
                'plots': {}
            }
        },
        'config': {
            '__init__.py': '',
        },
        'core': {
            '__init__.py': '',
        },
        'generators': {
            '__init__.py': '',
        },
        'agents': {
            '__init__.py': '',
        },
        'simulation': {
            '__init__.py': '',
        },
        'analysis': {
            '__init__.py': '',
        },
        'utils': {
            '__init__.py': '',
        },
        'scripts': {},
        'tests': {
            '__init__.py': '',
        },
        'gui': {
            '.gitkeep': '',
        }
    }
    
    def create_structure(base_path: Path, struct: dict):
        """Recursively create directory structure"""
        for name, content in struct.items():
            path = base_path / name
            
            if isinstance(content, dict):
                # It's a directory
                path.mkdir(exist_ok=True)
                print(f"📁 Created: {path}")
                create_structure(path, content)
            else:
                # It's a file
                if not path.exists():
                    path.write_text(content)
                    print(f"📄 Created: {path}")
    
    # Create from current directory
    base = Path('.')
    create_structure(base, structure)
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Data
data/output/*.csv
data/output/plots/*.png
data/personas/*.json
!data/input/.gitkeep
!data/personas/.gitkeep
!data/output/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
"""
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content)
        print(f"📄 Created: .gitignore")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Run the configuration files")
    print("2. Run the core modules")
    print("3. Run the generators")
    print("4. Run main.py")

if __name__ == "__main__":
    create_project_structure()