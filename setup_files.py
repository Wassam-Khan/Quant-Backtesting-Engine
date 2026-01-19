"""Setup script to create all necessary files"""
import os

files = [
    'src/core/config.py',
    'src/core/constants.py',
    'src/core/exceptions.py',
    'src/data/data_loader.py',
    'src/strategies/base_strategy.py',
    'src/strategies/golden_cross.py',
    'src/backtesting/backtester.py',
    'src/analytics/analyzers.py',
    'scripts/run_backtest.py',
    'tests/conftest.py',
    'config/config.yaml',
    'README.md',
    '.gitignore',
    'pytest.ini'
]

for file_path in files:
    file_path = file_path.replace('/', os.sep)
    
    # Get directory path
    dir_path = os.path.dirname(file_path)
    
    # Only create directory if it's not empty (i.e., not a root file)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('')
    
    print(f"✓ Created: {file_path}")

print("\n✅ All files created! Now fill them with content from the artifacts.")