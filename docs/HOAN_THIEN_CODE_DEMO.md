> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

# Hoàn thiện code demo - bản 2.0

## Kết quả hiện hành

- 26 URL demo độc lập: 24 trang trước đó và hai trang Q-learning.
- RAG sinh thực sự bằng Qwen2.5-1.5B-Instruct Q4_K_M, chạy CPU bằng llama.cpp trong thư mục dự án; nạp khi sử dụng.
- 27 nhóm kiểm tra tự động đạt, không skip; 197 artifact khớp SHA-256 trong lần kiểm tra.
- Ba yêu cầu sinh thực tế trên tiểu đường, Amazon và vàng đều HTTP 200; lưu câu hỏi, nguồn, model, câu trả lời và thời gian.
- Benchmark RL ba seed (42, 43, 44), lưu Q-table và log của từng cấu hình.

## Hai trang mới

### /rl/maze

Môi trường 8×8, bốn hành động, thưởng −1 mỗi bước, −8 va tường và +30 tới đích. Người dùng chỉnh episode, seed, alpha, gamma và epsilon. Xem đường Q-learning, đường BFS, chính sách từng ô, biểu đồ thưởng/số bước và phát lại đường đi.

Ở cấu hình mặc định, cả ba seed đạt đường 14 bước, bằng BFS. Đánh giá 100 lượt với Q đóng băng; random có tỷ lệ tới đích thấp hơn. Có thêm kiểm tra policy từ các ô xuất phát khác để tránh coi học được một đường là học được mọi ô.

### /rl/pricing

Thị trường mô phỏng: giá 10, vốn 5, giảm giá 20%, nhu cầu thấp/cao 10/20 khách, 20 bước/episode. Trạng thái gồm bước thời gian, nhu cầu, hành động trước của A và B. Agent B phản ứng theo luật công khai.

Q-learning, agent luật, luôn giảm giá và DP được đánh giá ở cùng vai A với cùng chuỗi nhu cầu và cùng ngẫu nhiên doanh số (CRN). Q không cập nhật khi đánh giá. DP biết mô hình và tối ưu tổng thưởng chiết khấu theo gamma; không coi nó là model học từ dữ liệu.

Đổi beta để so thị trường gốc và nhạy với giảm giá. Trên ba seed đã chạy, RL thua luật ở beta 0,15 và hơn luật ở beta 0,40; RL vẫn chưa đạt mốc DP. Đây là số đo trong mô phỏng, không đại diện thị trường thật. Khoảng xấp xỉ trong trang cho một model khác với SD giữa các seed trong benchmark.

Tải Q-table NPZ, log CSV hoặc toàn bộ JSON trực tiếp từ trang. Các run đặt ở `runs/rl/`, khóa bằng cấu hình và hash mã triển khai; chỉ nạp cache khi checksum Q-table còn khớp.

## RAG cục bộ

Runtime chính thức: llama.cpp b12284, gói Windows CPU x64. Model chính thức: Qwen/Qwen2.5-1.5B-Instruct-GGUF, revision `91cad51170dc346986eccefdc2dd33a9da36ead9`. Cả archive và GGUF được kiểm tra SHA-256 bởi script tải.

Tại `/knowledge/retrieval` hoặc ba trang KG, bật “Sinh bằng Qwen cục bộ”. Server LLM nghe ở localhost:8509, được khởi động khi cần. Chỉ cho một yêu cầu sinh đồng thời để tránh quá tải CPU. Nút dừng dự án kiểm tra PID/command line trước khi dừng cả web và LLM.

Đầu ra sinh dùng JSON schema; citation ID phải thuộc bằng chứng đã truy xuất. Điều này kiểm tra tính hợp lệ của mã nguồn, chưa chứng minh từng khẳng định có căn cứ. Ba ví dụ đã chạy:

| Câu hỏi | Kết quả thực tế |
|---|---|
| HbA1c phản ánh điều gì? | Mức đường huyết trung bình khoảng ba tháng trước, [K05] |
| AMZN Close ngày 2023-12-05? | 146.88, [S20231205] |
| Giá vàng ngày 2021-04-07? | 1738.05, [G20210407] |

Không coi ba ví dụ này là benchmark tổng quát chất lượng LLM. Corpus tiểu đường nhỏ; số liệu vàng chưa xác minh nguồn tải/đơn vị và vẫn giữ nguyên thang price.

## Hoàn thiện các demo đang có

- Sửa các input JSON sai kiểu có thể gây lỗi server; kiểm tra boolean và độ dài câu hỏi.
- Kiểm tra đầu ra phân bố xác suất và giá trị hữu hạn của model trước khi trả về API.
- Các kết quả ML/MLP, CNN và chuỗi mang thông tin artifact/hash để đối chiếu checkpoint thực sự được dùng.
- Kết quả API có run ID, trạng thái, timestamp và latency; RL có run ID và artifact riêng.
- Nhánh KG chấp nhận ngày ISO hoặc DD/MM/YYYY; hai ngày trong corpus cho thống kê trực tiếp các bản ghi, không để LLM tự tính số.
- Kiểm tra scope trước khi sinh các yêu cầu chẩn đoán/liều thuốc ngoài phạm vi.
- Giao diện tách từng trang; sửa bảng chương cho phù hợp sáu chương đã chọn; cập nhật cache key để tải JS mới.

## Kiểm chứng và tái lập

```powershell
python scripts/verify_demo.py --with-llm
python scripts/benchmark_rl.py
```

Các tệp chính:

- `runs/verification/test_summary.json`: kết quả test và checksum artifact.
- `runs/verification/real_rag_checks.json`: ba phản hồi model thực tế và evidence.
- `runs/verification/rl_benchmark.json`: chín cấu hình (ba maze, sáu pricing), từng run ID và tổng hợp.
- `assets/data/verified_extra_sequences.json`: kiểm chứng bốn checkpoint LSTM/GRU trước đó trên toàn bộ tập test.
- `scripts/setup_local_llm.py`: khôi phục runtime/model theo nguồn và hash cố định.

Xem README.md để chạy, dừng, mở từng trang hoặc chuyển máy. Các tài liệu phân tích tham khảo trước bản 2.0 có thể ghi RL/LLM là chưa có; trạng thái hiện hành được cập nhật ở tài liệu này.
