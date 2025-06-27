import os
from pathlib import Path

def create_project_structure():
    base_dir = "credit-risk-model"
    
    # Create directories
    dirs = [
        f"{base_dir}/.github/workflows",
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/notebooks",
        f"{base_dir}/src/api",
        f"{base_dir}/tests",
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create files
    files = {
        f"{base_dir}/.github/workflows/ci.yml": "",
        f"{base_dir}/notebooks/1.0-eda.ipynb": "",
        f"{base_dir}/src/__init__.py": "",
        f"{base_dir}/src/data_processing.py": "",
        f"{base_dir}/src/train.py": "",
        f"{base_dir}/src/predict.py": "",
        f"{base_dir}/src/api/__init__.py": "",
        f"{base_dir}/src/api/main.py": "",
        f"{base_dir}/src/api/pydantic_models.py": "",
        f"{base_dir}/tests/test_data_processing.py": "",
        f"{base_dir}/.gitignore": "",
        f"{base_dir}/Dockerfile": "",
        f"{base_dir}/docker-compose.yml": "",
        f"{base_dir}/requirements.txt": "",
        f"{base_dir}/README.md": "# Credit Risk Model\n\nProject description goes here.",
    }
    
    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created file: {file_path}")
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()