> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

> Cập nhật code: đã có 26 trang, hai demo RL và RAG Qwen cục bộ. Xem README.md và HOAN_THIEN_CODE_DEMO.md để biết trạng thái hiện hành. Nội dung bên dưới giữ ghi nhận của giai đoạn trước.

# Bổ sung demo chương 4–6

## Chương 4: SimpleRNN, LSTM và GRU

Mỗi kiến trúc và bài toán có một trang riêng:

| Bài toán | SimpleRNN | LSTM | GRU |
|---|---|---|---|
| Hành vi khách hàng | /rnn/customer | /lstm/customer | /gru/customer |
| Amazon | /rnn/amazon | /lstm/amazon | /gru/amazon |

SimpleRNN có Keras/PyTorch. LSTM và GRU dùng checkpoint **Keras** đã có trong `D:/A06_RNN/results`, không có checkpoint PyTorch tương ứng trong bộ demo. Không huấn luyện mới trong lần bổ sung giao diện này.

Đã tái tạo dữ liệu test từ CSV gốc, nạp lại bốn checkpoint và kiểm tra dự đoán trên toàn bộ tập test. Chi tiết máy đọc được ở `assets/data/verified_extra_sequences.json`. Hash dữ liệu và mốc chia tập của LSTM/GRU khớp A06: 140.000/30.000/30.000 chuỗi khách hàng và 1.542/337/338 mẫu cổ phiếu. Scaler chỉ fit train; đầu vào test lấy đúng các quan sát trước thời điểm mục tiêu.

| Mô hình | Accuracy khách hàng | Macro-F1 khách hàng | RMSE Amazon |
|---|---:|---:|---:|
| Keras SimpleRNN | 0,802967 | 0,526305 | 3,187731 |
| Keras LSTM | 0,810667 | 0,530884 | 3,870140 |
| Keras GRU | 0,814333 | 0,531864 | 3,108141 |

Baseline Amazon lấy giá phiên trước có RMSE **2,700446**, thấp hơn cả ba model. Không kết luận kiến trúc phức tạp luôn tốt hơn. Các checkpoint được chọn bằng validation trong script nguồn; đây là một lần chia tập và seed, không phải bằng chứng ưu thế tổng quát.

Script gốc được giữ ở `scripts/source_a06/compare_architectures.py` và `rnn_utils.py` để đối chiếu. Chúng giữ cấu trúc dữ liệu/kết quả của dự án A06_RNN gốc; không dùng nút dự đoán để huấn luyện.

## Chương 5: KG và truy xuất ba miền

| Trang riêng | Corpus | Nguồn |
|---|---|---|
| /knowledge/diabetes | 10 đoạn kiến thức và schema | WHO, NIDDK, schema BRFSS; có liên kết nguồn |
| /knowledge/stock | 250 phiên cuối AMZN | AMZN.csv từ A06 |
| /knowledge/gold | 250 ngày cuối có giá hợp lệ | D:/RNN/data/gold_price.csv |

Giữ giá trị gốc, không nội suy giá vàng. Nguồn CSV vàng không ghi đơn vị nên giao diện ghi rõ **đơn vị price của CSV, chưa xác định**; không gán USD/oz. Nguồn tải ban đầu của CSV vàng chưa được xác minh, vì vậy chỉ trình bày là demo trên file cục bộ, chưa coi là bộ dữ liệu đã xác nhận nguồn bên ngoài.

Ba phương pháp có thể chạy lại trực tiếp: KG khớp thực thể, TF-IDF ký tự 3–5, và Hybrid hợp nhất thứ hạng bằng RRF (k=60). Đồ thị có Document–MENTIONS–Entity; miền số có thêm Document–HAS_VALUE–Observation. Đồ thị trên màn hình là đồ thị của những tài liệu vừa truy xuất; số nút/cạnh corpus được tính từ toàn bộ miền.

Truy vấn số dùng ngày ISO YYYY-MM-DD để khóa bản ghi trước khi xếp hạng. Ngày không có trong corpus và câu hỏi giá hiện tại trả về không có bằng chứng. Do bước lọc ngày giải bài toán khóa bản ghi, không dùng điểm kiểm tra này để tuyên bố ưu thế ngữ nghĩa của TF-IDF/Hybrid.

**Đây là truy xuất và trả lời trích xuất có nguồn.** TF-IDF không phải embedding dense. Ba trang này chưa chạy sinh bằng LLM; không đổi nhãn thành kết quả RAG sinh. Trang `/knowledge/retrieval` có adapter Ollama cho kho tiểu đường; cần dịch vụ và model thực sự hoạt động. Máy hiện tại chưa xác nhận nhánh sinh, không có bảng đánh giá chất lượng LLM.

## Chương 6: Agent tích hợp

Trang `/agents/diabetes` cho xem Coordinator → Validation → Prediction → Knowledge → Synthesis tùy yêu cầu. Prediction gọi MLP BRFSS A03 thật; Knowledge truy xuất kho có nguồn. Hệ thống chạy tuần tự theo luật và tổng hợp theo mẫu. Không có agent tự lập kế hoạch bằng LLM, không có học tăng cường.

Trang `/experiments` chạy lại 21 câu hỏi truy xuất (7 mỗi miền), ba câu ngoài phạm vi và ba tuyến Agent: hỏi kiến thức, yêu cầu dự đoán thiếu hồ sơ và yêu cầu liều insulin ngoài phạm vi. Chi tiết ID mong đợi và ID tìm được hiện trên trang, có thể xuất JSON.

21 câu hỏi do người triển khai tự soạn để kiểm tra pipeline; không phải test set độc lập, không đại diện chất lượng hỏi đáp trên dữ liệu mới. Đánh giá tuyến Agent là kiểm tra xử lý yêu cầu, không phải đánh giá độ chính xác chẩn đoán.

## Đối chiếu phần còn lại của báo cáo

Chương 1 là tổng quan; chương 2 dùng các trang ML/MLP; chương 3 dùng các trang CNN đã có. Chương 7 tổng hợp kết quả, giới hạn và hướng phát triển, không cần một model mới. Khi viết báo cáo cần mô tả dataset/kiến trúc của chính bộ demo này: cổ phiếu là AMZN, chưa có FPT; mô hình Agent là BRFSS, không đổi thành dữ liệu UCI của bài mẫu.
