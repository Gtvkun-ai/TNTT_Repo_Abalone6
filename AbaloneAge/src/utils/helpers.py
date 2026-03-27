"""Tiện ích dùng chung trong dự án."""

from pathlib import Path
# Hàm ensure_directory giúp tạo thư mục nếu chưa tồn tại và trả về đối tượng Path,
#  đảm bảo rằng các thư mục cần thiết cho việc lưu trữ mô hình, kết quả hoặc các tệp khác được
#  tạo ra trong quá trình xử lý dữ liệu và huấn luyện mô hình đều tồn tại, giúp tránh lỗi liên
#  quan đến đường dẫn và đảm bảo rằng các tệp có thể được lưu trữ một cách an toàn và có tổ chức.
def ensure_directory(path: str | Path) -> Path:
    """Tạo thư mục nếu chưa tồn tại và trả về đối tượng Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
