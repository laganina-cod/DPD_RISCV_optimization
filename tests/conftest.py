import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работали импорты from src...
sys.path.insert(0, str(Path(__file__).parent.parent))