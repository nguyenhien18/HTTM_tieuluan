# Thực nghiệm bổ sung cho báo cáo HTTM

`robustness.py` là script đã thực sự chạy để tạo bảng nhiễu CNN trong Chương 3. Script giữ nguyên checkpoint, lấy 1.000 ảnh TEST phân tầng mỗi bộ, thử bốn mức nhiễu và ba seed nhiễu. Không huấn luyện hay chọn lại model. Kết quả đầy đủ, chỉ số mẫu, hash script và hash checkpoint nằm trong `data/robustness.json`.

`analyze_results.py` đọc các artifact của bản demo để tính thêm ma trận nhầm lẫn, Brier, phân tích nhóm và các thống kê báo cáo. Script không fit hoặc chọn lại mô hình. Kết quả tổng hợp nằm trong `data/evidence.json`; các CSV dẫn xuất nằm cùng thư mục dữ liệu.

Chạy bằng môi trường Python 3.13 của demo, đã cài requirements của dự án:

```powershell
cd D:\Tieuluanfinal
python Bao_cao\thuc_nghiem_bo_sung\analyze_results.py
python Bao_cao\thuc_nghiem_bo_sung\robustness.py
```

Hai script giữ `ROOT=Path('D:/Tieuluanfinal')` như lúc thực nghiệm. Nếu chuyển dự án sang đường dẫn khác, sửa duy nhất ROOT tương ứng rồi ghi lại hash mới; không nhận script đã sửa là byte-for-byte cùng script cũ. Script tạo kết quả cạnh chính nó trong thư mục `data/`; giữ bản kết quả nộp trước khi chạy lại nếu cần đối chiếu. Thời gian chạy có thể khác theo máy.

SD của bảng nhiễu chỉ phản ánh ba lần sinh nhiễu trên cùng mẫu và cùng model. Đây không phải SD giữa các lần huấn luyện hay khoảng tin cậy quần thể. Nguồn số liệu chính, run và giới hạn được trình bày đầy đủ trong báo cáo.
