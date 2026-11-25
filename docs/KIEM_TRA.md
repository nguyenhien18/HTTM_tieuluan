> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

> Cập nhật code: đã có 26 trang, hai demo RL và RAG Qwen cục bộ. Xem README.md và HOAN_THIEN_CODE_DEMO.md để biết trạng thái hiện hành. Nội dung bên dưới giữ ghi nhận của giai đoạn trước.

> Cập nhật: bộ demo đã có 24 trang. Phần bổ sung và kết quả mới xem CHUONG_4_5_6.md.

# Biên bản kiểm tra

Ngày kiểm tra: 11/11/2026. Máy hiện tại: Windows, Python 3.13, các phiên bản thư viện như requirements.txt.

## Kiểm tra tự động

10 bài kiểm tra unittest đã qua. Trong đó có vòng kiểm tra nhiều dataset/model:

- Trang chính và cấu hình đủ 5 bài toán bảng, 120 ảnh mẫu.
- 8 tổ hợp tabular: 3 ML A02, 3 MLP A03, 2 ML A01.
- Kraków xuất hiện trong category mà cả encoder A02 và A03 đã học, sau khi dùng mapping tương ứng.
- Từ chối hồ sơ thiếu trường, nhóm tuổi sai, số không hữu hạn và bài toán không tồn tại.
- 13 lựa chọn CNN trả vector 10 lớp hợp lệ: 5 MNIST, 4 Fashion-MNIST và 4 CIFAR-10.
- Từ chối ảnh/model không hợp lệ.
- Retailrocket chạy được Keras/PyTorch; không gán nhãn thật của mẫu nếu người dùng đã đổi chuỗi.
- Amazon Keras/PyTorch khớp CSV dự đoán đã lưu trên cả phiên đầu và cuối test trong sai số dưới 0,001.
- Truy xuất phân biệt type 1/type 2, ưu tiên đúng mục HbA1c; câu ngoài miền không có bằng chứng.
- Agent chạy tuyến kết hợp, thiếu hồ sơ trả need_input, câu yêu cầu liều thuốc trả out_of_scope.

## Kiểm tra trực tiếp trên trình duyệt

- Mở trang tổng quan, điều hướng các nhóm demo.
- Điền mẫu BRFSS và chạy so sánh Logistic Regression/MLP: có kết quả trực tiếp.
- MNIST ảnh mẫu: dự đoán và nhãn thật khớp.
- Vẽ số 7 bằng chuột trên canvas: model dự đoán số 7 và hiển thị ảnh đã xử lý.
- Retailrocket: chuỗi mẫu trả kết quả và nhãn thực tế để đối chiếu.
- Hỏi HbA1c: trả nội dung từ K05, nguồn NIDDK và đồ thị liên quan.
- Agent kết hợp hồ sơ và câu hỏi: hiển thị trace 7 bước.

## Phạm vi chưa xác nhận

- Chưa huấn luyện lại các model trong quá trình dựng giao diện.
- Chưa thử RAG sinh câu trả lời với Ollama vì máy chưa có dịch vụ/model Ollama hoạt động.
- Chưa có thử nghiệm chất lượng ngoài tập dữ liệu gốc hoặc nghiên cứu người dùng.
- Kiểm tra tự động và UI không làm thay đổi các giới hạn phương pháp của notebook gốc, ví dụ hồ sơ trùng BRFSS và một seed/một split.

## Chuyển sang trang độc lập

- 16 URL demo và trang danh mục trả HTTP 200; đường dẫn demo không tồn tại trả 404.
- Mỗi URL có title, tệp HTML và cấu hình cố định bài toán/model family.
- Không tải giao diện SPA cũ static/app.js; ML và MLP là các trang riêng, ba dataset CNN và hai bài toán RNN cũng tách riêng.
