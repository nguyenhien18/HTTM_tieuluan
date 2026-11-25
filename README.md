# HTTM Demo Studio 3.0 - code và thực nghiệm bàn giao

Ứng dụng có **26 trang riêng**, với ML/MLP được huấn luyện lại theo split nhóm, CNN/chuỗi được kiểm tra toàn bộ test, Qwen cục bộ và hai môi trường Q-learning. Dữ liệu, nguồn train/evaluate, checkpoint và kết quả được đóng gói trong dự án.

## Chạy

Nhấp đúp **CHAY_DEMO.cmd** trong thư mục dự án rồi mở http://localhost:8501. Nhấp **DUNG_DEMO.cmd** để dừng đúng máy chủ và model LLM của dự án. Có thể chạy `python launch.py` trong PowerShell.

Model Qwen được nạp khi bạn bật sinh câu trả lời lần đầu. Khi đã có runtime/model, suy luận hoạt động cục bộ, không cần API key. Chỉ nhánh sinh dùng LLM; truy xuất, ML/CNN/chuỗi và RL vẫn dùng đúng công cụ riêng.

## Kiểm tra bộ nộp bài

Nhấp **KIEM_TRA_NOP_BAI.cmd** hoặc chạy `python scripts/verify_submission.py --with-llm`. Mở [Kết quả thực nghiệm](http://localhost:8501/experiments) để xem split, baseline, log, confusion matrix, ảnh sai, các câu RAG chưa đạt và trace Agent. `passed` xác nhận các kiểm tra quy trình/chức năng đã định nghĩa; chất lượng model có bảng riêng.

Đọc [quy trình, dữ liệu và các lệnh tái lập](docs/QUY_TRINH_NOP_BAI.md). `TAI_LAP_THUC_NGHIEM.cmd` chạy lại/kiểm tra các nhóm bằng dữ liệu đóng gói; run hoàn tất cùng hash được kiểm tra và dùng lại.

```powershell
python -m experiments.tabular train --task all
python -m experiments.images evaluate --dataset all
python -m experiments.sequences
python -m experiments.rag
python -m experiments.agents
python scripts/verify_submission.py --with-llm
```

`data/raw/` chứa dữ liệu tabular/chuỗi, `data/images/` chứa snapshot ảnh. `experiments/` và `scripts/source_a06/` là bộ chạy offline; `runs/` lưu kết quả; `assets/models/submission/` chỉ định checkpoint tabular đang dùng. Không cần đường dẫn D:\Assignment* khi chạy các entry point hiện hành. Các notebook gốc lưu tại `experiments/legacy_sources/` để đối chiếu.

## Danh mục

| Trang | URL |
|---|---|
| ML · Tiểu đường BRFSS | [Mở](http://localhost:8501/ml/diabetes) |
| ML · Giá nhà Ba Lan | [Mở](http://localhost:8501/ml/house) |
| ML · Khả năng mua hàng | [Mở](http://localhost:8501/ml/ecommerce) |
| MLP · Tiểu đường BRFSS | [Mở](http://localhost:8501/mlp/diabetes) |
| MLP · Giá nhà Ba Lan | [Mở](http://localhost:8501/mlp/house) |
| MLP · Khả năng mua hàng | [Mở](http://localhost:8501/mlp/ecommerce) |
| CNN · Chữ số viết tay | [Mở](http://localhost:8501/cnn/mnist) |
| CNN · Phân loại trang phục | [Mở](http://localhost:8501/cnn/fashion-mnist) |
| CNN · Nhận diện ảnh màu | [Mở](http://localhost:8501/cnn/cifar10) |
| RNN · Hành vi khách hàng | [Mở](http://localhost:8501/rnn/customer) |
| RNN · Giá Amazon | [Mở](http://localhost:8501/rnn/amazon) |
| Knowledge Graph | [Mở](http://localhost:8501/knowledge/graph) |
| Hỏi đáp có nguồn | [Mở](http://localhost:8501/knowledge/retrieval) |
| Agent · Công cụ và kiểm định | [Mở](http://localhost:8501/agents/diabetes) |
| A01 · Tiểu đường | [Mở](http://localhost:8501/a01/diabetes) |
| A01 · Giá nhà Việt Nam | [Mở](http://localhost:8501/a01/house) |
| LSTM · Hành vi khách hàng | [Mở](http://localhost:8501/lstm/customer) |
| LSTM · Giá Amazon | [Mở](http://localhost:8501/lstm/amazon) |
| GRU · Hành vi khách hàng | [Mở](http://localhost:8501/gru/customer) |
| GRU · Giá Amazon | [Mở](http://localhost:8501/gru/amazon) |
| KG · Tiểu đường | [Mở](http://localhost:8501/knowledge/diabetes) |
| KG · Cổ phiếu Amazon | [Mở](http://localhost:8501/knowledge/stock) |
| KG · Giá vàng | [Mở](http://localhost:8501/knowledge/gold) |
| Kết quả thực nghiệm | [Mở](http://localhost:8501/experiments) |
| Q-learning · Mê cung | [Mở](http://localhost:8501/rl/maze) |
| Q-learning · Cạnh tranh giá | [Mở](http://localhost:8501/rl/pricing) |

## Cách dùng các phần mới

- **Mê cung:** chọn seed/episode/alpha/gamma/epsilon, huấn luyện, xem đường đi, hiện chính sách, so Q-learning với random và BFS. Có thể phát lại từng bước và tải Q-table/log CSV/kết quả JSON.
- **Cạnh tranh giá:** thay độ nhạy beta để xem môi trường gốc và môi trường đổi; so chính sách học với luật, luôn giảm giá và DP biết mô hình. Đánh giá dùng cùng nhu cầu/ngẫu nhiên doanh số; Q dừng cập nhật.
- **RAG:** trên Hỏi đáp hoặc ba trang KG, bật “Sinh bằng Qwen cục bộ”. Câu trả lời được sinh từ các nguồn vừa truy xuất; ID trích dẫn bị giới hạn trong danh sách nguồn. Mã nguồn hợp lệ chưa bảo đảm mọi khẳng định đều đúng, cần đối chiếu bằng chứng.
- **Hỏi số:** ba trang KG hỗ trợ ngày YYYY-MM-DD hoặc DD/MM/YYYY. Hai ngày trong corpus cho thống kê trực tiếp từ bản ghi; không để LLM tự tính số.
- **Xuất kết quả:** nút trên thanh đầu trang tải JSON của lần chạy. Các model suy luận trả thêm đường dẫn artifact và SHA-256.

## Kiểm chứng và chạy lại

Nhấp **KIEM_TRA_NOP_BAI.cmd** để kiểm tra toàn bộ bộ bàn giao. **KIEM_TRA_DEMO.cmd** là kiểm tra API/artifact nhanh. Hoặc:

```powershell
python scripts/verify_demo.py
python scripts/verify_submission.py --with-llm
python scripts/benchmark_rl.py
```

Kết quả nằm ở `runs/verification/`; từng lần huấn luyện RL ở `runs/rl/`. Cấu hình RL và mã triển khai giống nhau được dùng lại từ cache đã kiểm tra hash. Đổi seed/tham số tạo run khác.

## Chuyển máy hoặc khôi phục model LLM

Cài Python 3.13 và thư viện trong `requirements.txt`. Checkpoint scikit-learn/joblib cần môi trường tương ứng. Gói hiện tại chứa model/runtime LLM; nếu thiếu, chạy:

```powershell
python scripts/setup_local_llm.py
```

Script tải bản CPU chính thức llama.cpp b12284 và Qwen2.5-1.5B-Instruct Q4_K_M theo revision cố định, kiểm tra SHA-256 trước khi dùng. Model khoảng 1,12 GB; nguồn và hash nằm trong `runtime/llm_source.json`. Cổng web mặc định 8501; LLM chỉ nghe localhost ở 8509. `HTTM_PORT` và `HTTM_LLM_PORT` cho phép đổi cổng.

## Giới hạn cần giữ đúng khi trình bày

- Điểm phân loại BRFSS chưa hiệu chuẩn thành xác suất lâm sàng. Age là mã nhóm 1–13. Bộ demo cung cấp kết quả học tập, không chẩn đoán hay điều trị.
- ML/MLP hiện dùng cùng protocol và split nhóm; kết quả mới được lưu riêng. Các checkpoint CNN gốc vẫn có ngân sách khác nhau; chưa cô lập riêng ảnh hưởng framework.
- AMZN dự đoán một phiên lịch sử từ 30 phiên trước; baseline phiên trước hiện tốt hơn các checkpoint chuỗi đã lưu. Không phải dự báo thị trường hiện tại.
- Giá vàng giữ đơn vị `price` của CSV; nguồn tải/đơn vị chưa xác minh, không tự gán USD/oz.
- Đánh giá mới có 22 DEV/22 TEST tự soạn; 18 câu có bằng chứng được sinh thật và bốn câu ngoài phạm vi. Kiểm tra fact cố định và ID nguồn chưa thay thế đánh giá toàn diện chất lượng LLM.
- Agent ba miền điều phối theo luật, có Verifier/retry và benchmark lỗi kiểm soát. Hai demo RL dùng mô phỏng; không phải số đo trên thị trường thật.

Đề cương và phân tích tham khảo nằm trong `docs/`. Tình trạng code hiện hành và hướng dẫn chạy trong README này được ưu tiên khi các tài liệu phân tích cũ còn mô tả trạng thái trước khi bổ sung RL/LLM.
