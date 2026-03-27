"""Test configuration for resolving the project src package."""

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để có thể import các module từ src một cách dễ dàng trong quá trình viết test.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
