> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

> Cập nhật code: đã có 26 trang, hai demo RL và RAG Qwen cục bộ. Xem README.md và HOAN_THIEN_CODE_DEMO.md để biết trạng thái hiện hành. Nội dung bên dưới giữ ghi nhận của giai đoạn trước.

> Cập nhật: bộ demo đã có 24 trang. Phần bổ sung và kết quả mới xem CHUONG_4_5_6.md.

# Kịch bản trình diễn theo từng trang

Mở CHAY_DEMO.cmd, sau đó chọn từng thẻ trong danh mục. Mỗi URL có thể mở ở một tab riêng.

1. **ML tiểu đường**: /ml/diabetes. Chọn hồ sơ mẫu, chạy Logistic Regression.
2. **MLP tiểu đường**: /mlp/diabetes. Nhập cùng hồ sơ để xem kết quả mạng nơ-ron.
3. **CNN MNIST**: /cnn/mnist. Nhận diện ảnh mẫu, hoặc vẽ số 7 và chạy dự đoán. Có thể đổi NumPy/PyTorch/Keras trong đúng trang MNIST.
4. **CNN quần áo**: /cnn/fashion-mnist. Chọn ảnh trang phục rồi thử kiến trúc CNN.
5. **CNN ảnh màu**: /cnn/cifar10. Chọn ảnh CIFAR-10 và đối chiếu nhãn thật.
6. **RNN hành vi**: /rnn/customer. Chọn hoặc chỉnh chuỗi 5 hành vi; so sánh với nhãn thật khi giữ mẫu.
7. **RNN Amazon**: /rnn/amazon. Chọn phiên lịch sử, đối chiếu dự đoán và baseline.
8. **Knowledge Graph**: /knowledge/graph. Bấm vào nút để xem tài liệu nguồn của quan hệ.
9. **Hỏi đáp**: /knowledge/retrieval. Hỏi HbA1c, type 1/type 2; xem bằng chứng trả về. Mặc định trích xuất, chỉ bật RAG sinh trả lời khi Ollama sẵn sàng.
10. **Agent**: /agents/diabetes. Gửi kèm hồ sơ, quan sát trace 7 bước. Thử thiếu hồ sơ và yêu cầu ngoài phạm vi.

Giá nhà, mua hàng và hai demo A01 có thẻ riêng trong danh mục. Nút Xuất kết quả chỉ xuất lần chạy của trang đang mở. Đường dẫn của trang được giữ trong JSON.

Các chỉ số test là kết quả đã lưu của bài gốc. Không dùng khác biệt điểm trên một hồ sơ để khẳng định model tốt hơn. Model nhầm vẫn là ví dụ phân tích lỗi có ích.
