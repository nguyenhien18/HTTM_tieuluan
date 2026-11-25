> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

> Cập nhật: code hiện có 26 trang. Hai demo RL và nhánh RAG sinh đã được triển khai; các mục kế hoạch dưới đây cần đọc theo trạng thái mới trong README.md.

# Đề cương và quy ước báo cáo - dự án HTTM

## Hướng tổ chức đã chọn

Lấy cách tổ chức của bài tham khảo làm nền: lý thuyết, thiết kế, thực nghiệm, phân tích và triển khai. Nội dung, hình vẽ và kết quả phải phản ánh dữ liệu và mã nguồn của dự án này.

Báo cáo gồm lời mở đầu, sáu chương, kết luận, tài liệu tham khảo và phụ lục. Website tiếp tục có từng trang demo riêng cho từng bài toán và kiến trúc; các trang được gom theo chương khi trình bày báo cáo.

## 1. Đề cương báo cáo

| Chương | Nội dung chính | Thực nghiệm của dự án | Tình trạng |
|---|---|---|---|
| 1. Tổng quan và lịch sử AI | Các giai đoạn phát triển, ML/DL/tri thức/Agent, liên hệ với các chương sau | Timeline và sơ đồ tổng thể hệ thống | Cần biên soạn |
| 2. Machine Learning và MLP | Bài toán phân loại/hồi quy, tiền xử lý, thuật toán, mạng MLP, chỉ số đánh giá | BRFSS, giá nhà Ba Lan, khả năng mua hàng; ML A02 và MLP A03 | Đã có model và demo; cần hoàn thiện hình và phân tích |
| 3. CNN | Tích chập, pooling, kiến trúc, huấn luyện và phân tích lỗi | MNIST, Fashion-MNIST, CIFAR-10; các cấu hình A04/A05 | Đã có model và demo; cần tổng hợp hình từ kết quả thật |
| 4. RNN, LSTM và GRU | Chuỗi, cửa sổ thời gian, trạng thái ẩn, cơ chế cổng, baseline | Giá Amazon và hành vi Retailrocket; SimpleRNN, LSTM, GRU | Đã có checkpoint và demo; LSTM/GRU hiện dùng Keras |
| 5. Knowledge Graph và RAG | Biểu diễn tri thức, truy xuất, kết hợp phương pháp, sinh trả lời và đánh giá | Tiểu đường, AMZN và giá vàng; KG, TF-IDF, Hybrid | Truy xuất và RAG sinh bằng Qwen cục bộ đã chạy; dense retrieval chưa triển khai |
| 6. Agent và học tăng cường | Agent-môi trường, state/action/reward, Q-learning, agent luật và hệ phối hợp | Agent tiểu đường hiện có; dự kiến bổ sung mê cung và cạnh tranh giảm giá | Agent theo luật và hai demo Q-learning đã chạy; tiếp tục mở rộng đánh giá |

Kết luận tổng hợp phát hiện chính, giới hạn và hướng phát triển. A01 dùng làm ví dụ nhập môn hoặc phụ lục nếu cần; phần chính ưu tiên thực nghiệm có đủ dữ liệu để phân tích.

## 2. Format chung cho một bài toán thực nghiệm

1. **Mục tiêu:** đầu vào, đầu ra, bài toán cần giải quyết và câu hỏi muốn kiểm chứng.
2. **Dữ liệu:** nguồn, phiên bản, kích thước, trường dữ liệu, đơn vị, phân bố, thiếu và trùng lặp.
3. **Thiết kế:** chia train/validation/test, tiền xử lý, baseline, kiến trúc và sơ đồ có kích thước đầu vào/đầu ra.
4. **Cài đặt:** đoạn code cốt lõi, bảng siêu tham số, seed, môi trường chạy và quy tắc chọn checkpoint.
5. **Kết quả:** bảng chỉ số, biểu đồ phù hợp, trường hợp đúng/sai và đối chiếu baseline.
6. **Phân tích:** hiện tượng quan sát được, bằng chứng, cách giải thích và giới hạn của kết luận.
7. **Demo:** một tình huống có đầu vào, đầu ra và mô hình/nguồn được dùng; kèm ảnh giao diện.
8. **Tiểu kết:** trả lời câu hỏi thực nghiệm đã đặt ra ở đầu mục.

Không bắt buộc mọi chương có cùng số hình hoặc số trang. Chọn hình giúp người đọc hiểu dữ liệu, cơ chế hoặc kết quả.

## 3. Tổ chức mã nguồn

Các phần đang có:

| Vị trí | Vai trò |
|---|---|
| `app.py` | Route và API của ứng dụng |
| `demo/` | Nạp model, kiểm tra dữ liệu, suy luận, truy xuất, điều phối và danh mục trang |
| `templates/pages/`, `static/pages/` | Giao diện riêng của từng demo |
| `assets/data/` | Dữ liệu phục vụ demo, metadata và kết quả đã tổng hợp |
| `assets/models/` | Checkpoint và bộ tiền xử lý |
| `assets/manifest.json` | Nguồn và mã băm của artifact |
| `scripts/` | Script hỗ trợ và nguồn thực nghiệm đã lưu |
| `tests/` | Kiểm tra model/API/truy xuất và lỗi đầu vào |
| `docs/` | Hướng dẫn, đề cương và ghi nhận kiểm chứng |

Khi bổ sung phần thực nghiệm có thể tái lập, tổ chức theo cấu trúc sau. Đây là **cấu trúc đích**, không phải khẳng định các thư mục hoặc script này đã tồn tại:

```text
experiments/
  ch02_tabular/
  ch03_cnn/
  ch04_sequences/
  ch05_kg_rag/
  ch06_agents_rl/
results/
  ch02/ ... ch06/
report/
  chapters/
  figures/
  tables/
```

Mỗi nhóm thực nghiệm có cấu hình, script hoặc notebook, hướng dẫn chạy và kết quả máy đọc được. Code nạp dữ liệu/model dùng chung được đặt trong module chung; notebook dùng để trình bày và gọi các module đó. Bước huấn luyện, đánh giá và suy luận được phân biệt rõ trong lệnh chạy.

## 4. Liên kết code, số liệu, sơ đồ và báo cáo

Chuỗi truy vết mong muốn:

```text
Dữ liệu + cách chia tập + cấu hình
    → huấn luyện và chọn checkpoint
    → đánh giá TEST và lưu dự đoán
    → sinh bảng/biểu đồ
    → viết phân tích trong báo cáo
    → demo dùng cùng checkpoint và preprocessing
```

Một thực nghiệm hoàn chỉnh cần lưu được các thông tin sau, ở một hoặc nhiều tệp:

- Dữ liệu nguồn và hash; seed, mốc chia tập và danh sách/khóa mẫu khi thích hợp.
- Cấu hình model, preprocessing, ngân sách huấn luyện và tiêu chí chọn checkpoint.
- Trọng số đã chọn, lịch sử huấn luyện nếu có, dự đoán TEST và chỉ số tính từ dự đoán.
- Script sinh hình/bảng và metadata đo thời gian.
- Ví dụ đầu vào/đầu ra để đối chiếu giữa notebook, API và báo cáo.

## 5. Quy ước hình vẽ và phân tích

- Sơ đồ kiến trúc phải theo model thật: MLP hiện có gồm bốn tầng ẩn 64-32-16-8 và tầng đầu ra; không dùng sơ đồ sáu lớp học được của bài mẫu.
- Sơ đồ hệ thống ghi rõ phần đang chạy và phần đề xuất. Nhánh LLM chưa xác nhận không được trình bày như một kết quả thực nghiệm đã hoàn thành.
- Hình dữ liệu lấy từ CSV/ảnh của dự án. Biểu đồ kết quả lấy từ log, prediction hoặc metrics đã lưu.
- Confusion matrix và ảnh dự đoán sai dùng TEST; đường loss cần phân biệt TRAIN/VALIDATION. Không suy ra đường loss nếu chưa có log.
- Bảng so sánh ghi dataset, split, metric, model và điều kiện chạy. Chỉ thêm trung bình/độ lệch chuẩn khi đã chạy nhiều seed.
- Mỗi hình có số, tên, nhãn trục/đơn vị nếu có và đoạn phân tích. Đoạn phân tích nêu số đo trước, giải thích sau, rồi giới hạn.
- Giữ cả kết quả model thua baseline. Trên Amazon, baseline phiên trước đang có RMSE tốt hơn các checkpoint chuỗi đã kiểm tra; đây là nội dung cần phân tích.

## 6. Những khác biệt phải giữ đúng khi viết

- BRFSS A02/A03 của dự án là phân loại nhị phân; `Age` là mã nhóm tuổi, không phải số tuổi trực tiếp và không phải nhãn type 1/type 2.
- Cổ phiếu đang dùng Amazon; Retailrocket dự đoán sự kiện tiếp theo, không phải doanh thu Online Retail.
- A02 refit train+validation, A03 giữ train gốc; các cấu hình MNIST cũng có ngân sách huấn luyện khác nhau. Chênh lệch kết quả hiện có chưa cô lập riêng ảnh hưởng thuật toán/framework.
- Giá vàng dùng thang `price` của CSV cục bộ, đơn vị và nguồn tải ban đầu chưa xác minh. Cần hoàn thiện provenance trước khi dùng làm thực nghiệm chính trong báo cáo.
- Bộ 21 câu hỏi KG hiện tại là kiểm tra tự soạn. Tra cứu theo ngày trong CSV không thay thế một benchmark hỏi đáp ngữ nghĩa độc lập.
- Agent hiện có điều phối theo luật và tổng hợp theo mẫu. Q-learning cần môi trường, hành động, reward và quá trình học riêng.

## 7. Thứ tự hoàn thiện

1. Chuẩn hóa đầu ra thực nghiệm và tạo các bảng/hình còn thiếu cho ML, CNN và chuỗi đã có.
2. Hoàn thiện corpus, nguồn dữ liệu và đánh giá KG/RAG; chạy bộ sinh nếu đưa kết quả LLM vào báo cáo.
3. Bổ sung, kiểm chứng hai thí nghiệm RL; đánh giá hệ Agent theo nhóm yêu cầu và trường hợp lỗi.
4. Viết các chương theo format thống nhất, chèn sơ đồ/biểu đồ đã kiểm tra, rồi chụp các demo tiêu biểu.
5. Đối chiếu toàn bộ số liệu, cấu hình, hình, đường dẫn và kết luận trước khi xuất bản báo cáo.

Đây là đề cương làm việc của dự án. Các hạng mục ở trạng thái cần bổ sung chỉ trở thành kết quả báo cáo sau khi đã triển khai và kiểm chứng.


## 8. Bổ sung sau khi đối chiếu bốn báo cáo khác

Phân tích chi tiết và thứ tự ưu tiên xem [DOI_CHIEU_4_BAO_CAO.md](DOI_CHIEU_4_BAO_CAO.md). Các mục dưới đây là đề xuất cần triển khai, chưa phải chức năng mới đã hoàn tất:

1. Một registry thí nghiệm và script sinh hình/bảng từ kết quả thật; liên kết dataset, split, metric, cấu hình và checkpoint.
2. Gom kiểm chứng artifact/preprocessing/đầu ra tham chiếu thành lệnh chạy lại trong dự án.
3. Thêm kiến trúc tổng thể, sequence diagram và schema trạng thái/công cụ vào phần thiết kế.
4. Đánh giá PR, calibration và ngưỡng quyết định cho phân loại; mọi lựa chọn ngưỡng/hiệu chuẩn dùng validation phù hợp.
5. Thử độ bền ảnh với các mức nhiễu được xác định trước; phân tích sai số chuỗi theo giai đoạn và giữ baseline.
6. Phân biệt các công cụ đọc KG, tính số trên CSV và truy xuất/sinh văn bản; đầu ra truy được nguồn và công cụ thực sự đã dùng.
7. Đánh giá Agent trên nhóm yêu cầu và lỗi công cụ; tách thành công điều phối khỏi độ chính xác dự đoán.
8. Xem thí nghiệm trí nhớ RNN/LSTM/GRU, Agent dùng LLM, SQLite và ONNX là các mở rộng có mục tiêu riêng; không tự động đưa tất cả vào phạm vi bắt buộc.

Giữ khung sáu chương và từng demo một trang. Những bổ sung này tăng chiều sâu đánh giá và khả năng tái lập; các hạng mục RAG sinh và RL đã ghi trước đó vẫn giữ đúng trạng thái hiện tại.
