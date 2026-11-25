# Quy trình thực nghiệm và bàn giao demo HTTM

## Phạm vi bản hiện hành

26 trang demo được giữ riêng. Phần offline và phần suy luận dùng chung dữ liệu, cấu hình và checkpoint có thể truy vết. Bản này sửa các thiếu hụt được ghi trong lần audit trước, không lấy số liệu thực nghiệm từ bài tham khảo của sinh viên khác.

Các phần đã thực hiện:

1. Đóng gói dữ liệu gốc tabular, AMZN, Retailrocket và snapshot train/test của ba bộ ảnh.
2. Huấn luyện lại ML/MLP của ba bài toán bằng một protocol dùng chung cho từng bài; chia theo nhóm feature để nhóm không giao nhau.
3. Chọn cấu hình ML và epoch MLP bằng validation; giữ nguyên train cho cả hai, không refit thêm validation ở một phía.
4. Nạp lại sáu checkpoint tabular trên toàn bộ test, đối chiếu prediction và chỉ số.
5. Đánh giá lại 13 checkpoint CNN trên 10.000 ảnh test/model, lưu xác suất, confusion matrix, classification report và ảnh dự đoán sai.
6. Sửa đường dẫn bộ chạy chuỗi, đóng gói nguồn/metadata/log; chạy lại cả tám checkpoint trên 338 phiên AMZN và 30.000 chuỗi Retailrocket/model.
7. Thêm BM25 và đường đi quan hệ KG, benchmark 22 DEV/22 TEST tự soạn; thực sự sinh 18 câu có bằng chứng và kiểm tra bốn câu cần từ chối.
8. Thêm Agent đa miền theo luật, Verifier, retry một lần, kiểm tra tuyến và đối chứng tiêm lỗi công cụ.
9. Giữ hai môi trường Q-learning, baseline và benchmark ba seed.
10. Cung cấp lệnh kiểm tra, lệnh tái lập và manifest bàn giao.

## 1. Cách chạy và kiểm tra

Mở terminal trong thư mục dự án. Máy đã có Python và thư viện tương thích có thể chạy ngay:

```powershell
python launch.py
python scripts/verify_submission.py --with-llm
```

Hoặc nhấp `CHAY_DEMO.cmd`, `KIEM_TRA_NOP_BAI.cmd`. Kết quả tổng hợp có ở `runs/verification/submission.json`, hiển thị tại `/experiments`. `DUNG_DEMO.cmd` dừng tiến trình web và model LLM thuộc đúng dự án.

Trên máy khác: cài Python 3.13, chạy `python -m pip install -r requirements.txt`. Đây là bản Windows CPU; runtime llama.cpp đi kèm là Windows x64. Không cần API key. Nếu thiếu model/runtime, dùng `python scripts/setup_local_llm.py`; bước này mới cần mạng. Giải nén cả dữ liệu và runtime khi chuyển máy.

## 2. Dữ liệu và nguồn

| Bài toán | Tệp bàn giao | Nhãn/đầu ra | Ghi nhận nguồn |
|---|---|---|---|
| BRFSS | `data/raw/diabete.csv` | `Diabetes_binary`, 21 feature | Bản A02/A03, link gốc notebook: Kaggle alexteboul/diabetes-health-indicators-dataset |
| Nhà Ba Lan | `data/raw/house.csv` | `price`, PLN | Bản A02/A03, link notebook: Kaggle g1llar/poland-olx-house-price-q122 |
| E-commerce | `data/raw/customerData.csv` | `PurchaseStatus`, 16 feature | Bản 500.000 dòng/17 cột trong A02/A03; link notebook gauthamvijayaraj/customer-purchase-behavior-dataset-e-commerce chưa chứng minh cách tạo đúng tệp này |
| Amazon | `data/raw/AMZN.csv` | Close lịch sử | Bản A06; hash đối chiếu metadata |
| Retailrocket | `data/raw/events.csv` | Hành vi tiếp theo | Bản A06; tạo chuỗi trong cùng visitor, chỉ dùng sự kiện trước target |
| MNIST/Fashion/CIFAR-10 | `data/images/*.npz` | 10 lớp ảnh | Snapshot từ các tệp/thư mục của A04/A05, giữ train/test nguồn và class order; JSON kèm hash và mô tả |
| Giá vàng | `assets/data/gold_price.csv` | `price` lịch sử | Giữ tên thang giá nguồn cục bộ; nhà cung cấp/đơn vị gốc chưa xác minh, dùng làm tra cứu phụ |

`data/sources.json` ghi hash/byte và bản sao nguồn đã dùng. URL được lấy từ notebook cũ; không tự coi đó là bằng chứng tệp cục bộ khớp hoàn toàn một phiên bản Kaggle công khai. Mọi phép tính tái lập trong bản nộp dùng tệp bàn giao có hash.

Không dùng `price_per_meter` hoặc tiêu đề tin để dự đoán giá nhà. Chuẩn hóa city dùng cùng bảng ánh xạ cho train và inference. E-commerce đổi bốn nhóm giá trị âm bất hợp lệ thành missing, sau đó imputer chỉ fit trên train. BRFSS Age là mã 1–13.

## 3. ML/MLP: chuẩn bị, huấn luyện và đánh giá

```powershell
python -m experiments.tabular train --task all
python -m experiments.tabular evaluate --run runs/tabular/diabetes/9b3d1d998e86
```

Lệnh train dùng lại run đã hoàn tất nếu dữ liệu, protocol và mã nguồn có cùng hash, đồng thời kiểm tra tính toàn vẹn. Trên bản không có run đó, lệnh thực hiện huấn luyện. Muốn tạo thí nghiệm cấu hình khác, sửa protocol có chủ đích; hash tạo run mới. Giữ checkpoint cũ làm tham chiếu, không sửa metric thủ công.

Mỗi nhóm feature giống nhau, kể cả trường hợp nhãn khác nhau, được giữ trong một split. Seed 42; tỷ lệ gần 70/15/15 theo nhóm, số dòng thực tế được ghi trong `split.json` và từng hàng trong `split.csv`.

- Tiền xử lý số: imputer median và StandardScaler; phân loại: imputer và OneHotEncoder, fit trên TRAIN.
- ML phân loại: Logistic Regression với ba mức C, có class weight; chọn bằng F1 validation.
- ML giá nhà: Ridge và Random Forest; chọn bằng RMSE validation.
- MLP: bốn tầng ẩn 64–32–16–8 và một tầng đầu ra; Adam; tối đa 35 epoch, patience 6; checkpoint theo F1 hoặc RMSE validation. Target giá nhà được chuẩn hóa bằng thống kê TRAIN.
- Baseline: prior class hoặc median giá từ TRAIN. Không dùng TEST để chọn model, threshold hoặc epoch. Chưa hiệu chuẩn xác suất lâm sàng; threshold 0,5 cố định.

Mỗi run có config, dataset metadata, split, preprocessing, trọng số, bảng validation, selection, log, prediction TEST, metric, ví dụ lỗi và hình. Cả ML/MLP dùng đúng cùng train/validation/test. Demo đọc run được chỉ định trong `assets/models/submission/<task>/registry.json`; response trả tên run và hash artifact.

Đã nạp lại model/preprocessor để tái tính toàn tập TEST. Chưa chạy nhiều seed tabular, nên không báo sai số giữa seed. Các số mới không được ghép với số A02/A03 cũ dưới cùng tên cấu hình.

## 4. CNN

```powershell
python -m experiments.images evaluate --dataset all
python -m experiments.images train --dataset fashion --model basic_cnn --epochs 5
python -m experiments.trained_images --run runs/cnn_training/fashion/basic_cnn/<run>
```

Snapshot đóng gói đã đủ để chạy, không cần đường dẫn A04/A05. Lệnh `prepare` chỉ dùng khi xây lại snapshot từ tệp nguồn khác.

Đánh giá chính dùng 13 checkpoint gốc đã kiểm chứng. Mỗi run ghi checkpoint hash, dataset hash, prediction của 10.000 ảnh, xác suất các lớp, metric, confusion matrix và 16 ảnh sai có confidence cao nhất. Thứ tự lớp được lưu rõ. Ảnh tải lên bất kỳ không được coi là mẫu test cùng phân phối.

Lệnh train tạo run mới, giữ test nguồn và tách 20% train làm validation, chọn checkpoint bằng validation loss. Có nhánh PyTorch, Keras và NumPy scratch cho MNIST. Keras dựng model mới từ cấu hình kiến trúc trong checkpoint mẫu, không dùng lại trọng số đã học. NumPy có forward/backward từ A04; phép Conv2D/Pool được đối chiếu autograd, softmax loss được kiểm tra bằng sai phân hữu hạn.

Đã thử đường train/evaluate mới với MNIST PyTorch và Fashion Keras một epoch để kiểm tra bộ chạy. Đây là các run kiểm chứng kỹ thuật, không tự thay thế checkpoint demo chính hoặc dùng để kết luận chất lượng kiến trúc.

Các model gốc có điều kiện huấn luyện khác nhau. Bảng chính so các cấu hình đã thực hiện, không kết luận nguyên nhân chênh lệch chỉ do framework. Chưa có phép kiểm chứng đồng nhất initialization/batch/optimizer trên cả mạng như mức đối chứng nâng cao của bài mẫu.

## 5. Chuỗi

```powershell
python scripts/source_a06/compare_architectures.py --prepare-only
python -m experiments.sequences
python scripts/source_a06/compare_architectures.py --task stock --architecture gru
```

Entry point mặc định đọc `data/raw/`, ghi run mới ở `runs/sequences/retrained/`; có `--data-dir` và `--output-dir`. Hỗ trợ Keras SimpleRNN/LSTM/GRU. Nguồn PyTorch và notebook gốc có ở `experiments/legacy_sources/`; những notebook lưu trữ có đường dẫn lịch sử, dùng để truy vết thay vì coi là entry point bàn giao.

AMZN: lọc từ 2015, dùng 30 Close trước target, scaler fit phần train, chia thời gian 70/15/15. Retailrocket: lịch sử năm hành vi trong cùng visitor; loại timestamp không bảo đảm quá khứ; lấy mẫu theo thời gian không dựa vào nhãn; class weight từ train; split không cắt timestamp bằng nhau.

Cả tám checkpoint đã chạy lại toàn bộ test và so với prediction lưu. Có actual/prediction, baseline, confusion matrix hoặc residual. Các checkpoint AMZN hiện có RMSE cao hơn persistence; giữ kết quả này để phân tích. Không gọi đây là dự báo giá hiện tại hay dự báo nhiều bước.

## 6. KG/RAG

```powershell
python -m experiments.rag
```

Kho kiến thức nhỏ, có nguồn. TF-IDF, BM25 và KG là ba đối chứng; Hybrid dùng RRF k=60. KG ghi đường đi tối đa hai cạnh trên quan hệ có source ID. Đây là truy xuất bằng đường đi, chưa phải suy luận tri thức tổng quát.

`experiments/fixtures/rag_cases.json` có 22 DEV và 22 TEST, câu hỏi tách biệt. Tham số và phương pháp Hybrid được xác định trước khi chạy TEST. Cùng KB là điều bình thường trong bài truy xuất; đây vẫn là bộ câu tự soạn, không phải benchmark bên ngoài độc lập. Giữ riêng nhóm câu khái niệm và lookup khóa ngày.

Đã sinh thật 10 câu khái niệm và tám câu số; bốn câu ngoài phạm vi không được gửi cho bộ sinh. Báo Hit@3/MRR@3 cho truy xuất, token F1/ROUGE-L, kiểm tra fact cố định, source ID và thời gian cho sinh. Các câu không đạt được giữ nguyên trong prediction JSON. ID hợp lệ và fact check không chứng minh đầy đủ tính đúng của mọi câu văn. Không sửa luật theo TEST rồi vẫn gọi TEST đó là đánh giá độc lập.

## 7. Agent và RL

```powershell
python -m experiments.agents
python scripts/benchmark_rl.py
```

Agent theo luật có ba miền dự đoán: tiểu đường, nhà Ba Lan và AMZN. Router/Validation/Specialist/Verifier/Synthesis có trách nhiệm và trace rõ; kiểm tra output trước khi tổng hợp, thử lại tối đa một lần, dừng khi lỗi kéo dài. Trạng thái tool_error không giả làm kết quả dự đoán.

Benchmark có 17 tình huống tuyến/đầu vào và 30 trường hợp tiêm lỗi cho mỗi biến thể basic/verified. Đây là lỗi kiểm soát timeout/NaN/ngoài miền, không phải tỷ lệ dự đoán đúng bệnh/giá. API không cho người dùng bật fault hook.

RL có mê cung và định giá mô phỏng. Q được đóng băng khi đánh giá. Mê cung so random/BFS và ghi khả năng đi từ các ô khác; định giá so cùng vai/cùng đối thủ/cùng ngẫu nhiên doanh số, có DP biết mô hình. Có ba seed huấn luyện. Phân biệt SD giữa seed và giữa episode. Mê cung vẫn là một bản đồ cố định; giá mô phỏng không đại diện lợi nhuận thị trường thực.

## 8. Tái lập, kiểm thử và truy vết

```powershell
python scripts/reproduce.py --train-tabular
python scripts/export_architectures.py
python scripts/verify_submission.py --with-llm
```

Lệnh reproduce kiểm tra/tái tạo từng nhóm bằng dữ liệu đã đóng gói. Run trùng đã hoàn tất có thể được kiểm tra hash và dùng lại; nó không giả tạo một lần huấn luyện mới. Muốn kiểm tra từ đầu, dùng một bản sao dự án không có các run tương ứng và giữ bộ bàn giao gốc để đối chiếu.

`runs/verification/submission_manifest.json` băm code, dữ liệu, model, nguồn notebook và kết quả bàn giao. `complete.json` của từng run băm các artifact của run đó. `submission.json` tổng hợp kiểm tra đã thực hiện và các giới hạn. Việc passed chỉ nói các kiểm tra định nghĩa đã qua, không chứng minh mọi model/LLM đều trả lời đúng.

`python scripts/check_package.py` đối chiếu manifest bàn giao mà không ghi lại manifest. File CMD kiểm tra nộp bài chạy bước này trước. Khi chủ động sửa code hoặc tạo bộ bàn giao mới, dùng trực tiếp `verify_submission.py` để kiểm tra và xuất manifest mới; không dùng thao tác đó để che một thay đổi chưa rõ nguồn gốc.

`assets/data/architecture_registry.json` xuất shape/số tham số từ model thật, dùng làm nguồn vẽ sơ đồ. Nội dung notebook cũ được giữ trong `experiments/legacy_sources/` để đối chiếu; README và các entry point hiện hành là hướng dẫn chạy.
