> Tài liệu từ giai đoạn trước. Bản code 3.0 đã bổ sung pipeline, split nhóm, đánh giá toàn tập, RAG/Agent và kiểm tra bàn giao. Trạng thái và lệnh hiện hành xem [QUY_TRINH_NOP_BAI.md](QUY_TRINH_NOP_BAI.md).

# Đối chiếu bốn báo cáo và đề xuất nâng chất lượng dự án HTTM

## Phạm vi và kết luận

Đối chiếu cấu trúc, các phần phương pháp, thiết kế, thực nghiệm, giới hạn và những trang hình tiêu biểu của bốn PDF với code/demo hiện tại trong `D:/Tieuluanfinal`. Các số liệu của báo cáo tham khảo dưới đây được xem là kết quả tác giả trình bày, chưa được xác nhận độc lập bằng mã nguồn của họ.

| Bài tham khảo | Tệp nguồn | Số trang PDF |
|---|---|---:|
| Phan Văn Khôi | D:/Downloads/tieuluan_final_CT_03_khoipv.447.pdf | 132 |
| Phan Văn Hoàng Việt | D:/Downloads/tieuluanfinal_CT01_09_vietpvh.pdf | 96 |
| Nguyễn Hữu Khánh Duy | D:/Downloads/tieuluanfinal_06_04_duynhk.pdf | 105 |
| Đinh Hải Triều | D:/Downloads/TieuLuan_final_06_14_trieu.843.PDF | 239 |

**Quy ước:** mọi số trang dẫn trong tài liệu này là thứ tự trang PDF, bắt đầu từ 1, để tránh nhầm với số trang in của từng báo cáo.

Giữ khung sáu chương đã chọn. Những bổ sung có ích nhất là thiết kế hệ thống có hợp đồng dữ liệu rõ ràng; đánh giá mất cân bằng và độ tin cậy của điểm dự đoán; thí nghiệm độ bền; kiểm chứng checkpoint và kết quả API; hỏi đáp có nguồn phân biệt truy vấn số với truy xuất văn bản; và đánh giá Agent theo kịch bản. Chưa cần mở rộng số dataset hay chuyển toàn bộ website sang công nghệ khác.

Đây là **bản phân tích và danh mục nâng cấp đề xuất**, không phải thông báo các chức năng mới đã được lập trình.

## 1. Bài Khôi: sát với dữ liệu và luồng demo hiện tại

### Những phần có thể học theo

- **Ngưỡng quyết định được chọn trên validation.** Trang PDF 31–32 trích quy trình dự đoán điểm trên validation, chọn threshold rồi dùng threshold đó để đánh giá TEST. Đây là hướng hữu ích cho BRFSS/e-commerce, nơi Accuracy và F1/Recall phản ánh các khía cạnh khác nhau.
- **Phân biệt benchmark pipeline và benchmark framework.** Trang 49 ghi rõ các model cùng TEST nhưng khác số ảnh TRAIN; thời gian suy luận theo batch trong notebook khác độ trễ một yêu cầu localhost. Rất phù hợp với A04 của bạn, vì NumPy/Keras/PyTorch chưa cùng kiến trúc và ngân sách huấn luyện.
- **EDA gắn với kiểu dữ liệu.** Nhãn và missing values cho bảng; ảnh mẫu và nhầm lẫn theo lớp cho CNN; rolling statistics, actual–predicted và phần dư cho chuỗi. Có thể bổ sung các hình tương ứng trên chính dữ liệu của bạn.
- **Ba công cụ trả lời khác nhau.** Trang 89–91 phân biệt đọc graph cho một bản ghi/nhóm, tính CSV cho thống kê và retrieval + LLM cho mô tả. Kết quả mang `answer`, `sources`, `tools`, giúp biết con số hoặc câu trả lời được tạo bằng cách nào.
- **Ảnh demo phải có kết quả đã chạy.** Trang 85 nhấn mạnh ảnh chỉ có biểu mẫu chưa chứng minh suy luận thành công. Ảnh đưa vào báo cáo nên thấy đầu vào, lựa chọn model và đầu ra.

### Áp dụng vào dự án

1. Giữ endpoint hiện có, bổ sung metadata cho phép truy vết tool/model/checkpoint của kết quả.
2. Với hỏi đáp AMZN: một ngày đọc bản ghi; khoảng ngày dùng phép tính có kiểm soát; câu hỏi mô tả truy xuất tài liệu. Mỗi nhánh có kiểu nguồn và giới hạn riêng.
3. Nếu tối ưu threshold, lưu threshold cùng model và tiêu chí lựa chọn. Không chọn threshold bằng TEST hoặc lấy threshold của bài tham khảo.
4. Ghi hai loại thời gian: nạp model lần đầu và các lần suy luận sau khi đã nạp; tách thời gian model khỏi tổng thời gian API khi có thể.

### Điểm cần giữ giới hạn

TF-IDF là biểu diễn từ vựng/ký tự, không phải dense semantic embedding. Có ID nguồn nghĩa là đã truy xuất tài liệu đó, chưa chứng minh mọi câu trong đầu ra LLM được nguồn hỗ trợ. Dự báo một bước cũng không tự thành dự báo nhiều bước.

## 2. Bài Việt: bổ sung thiết kế phần mềm và phân tích mô hình

### Những phần có thể học theo

- **Thiết kế trước khi mô tả triển khai.** Chương 3 có yêu cầu chức năng/phi chức năng, use case, sequence diagram, đồ thị trạng thái, class diagram và ontology. Các trang PDF 38–46 cho thấy từng sơ đồ trả lời một câu hỏi về hệ thống.
- **Vai trò Agent có đầu vào, công cụ và đầu ra.** Bảng 3.1, trang 38–39, hữu ích hơn một danh sách tên Agent. Nên áp dụng cho Coordinator, Validation, Prediction, Knowledge và Synthesis hiện có.
- **Trạng thái có cấu trúc.** Trang 41 mô tả AgentState chứa dữ liệu đầu vào, dự đoán, bằng chứng và log. Có thể áp dụng bằng typed structure/dataclass/schema, không bắt buộc dùng LangGraph.
- **Đánh giá ngoài Accuracy.** Các trang 53–59 có ROC, PR, confusion matrix, calibration/Brier và SHAP. Đặc biệt đáng cân nhắc cho BRFSS mất cân bằng.
- **Phân tích trường hợp và thời gian từng bước.** Trang 76, 79–81 trình bày các tình huống end-to-end, sự khác nhau giữa hai nguồn đánh giá và thời gian thành phần.

### Áp dụng vào dự án

1. Thêm ba hình chính vào báo cáo: kiến trúc tổng thể; sequence diagram một yêu cầu có gọi model và truy xuất; schema đầu vào/đầu ra/trạng thái.
2. Đặc tả bảng Agent: vai trò, input bắt buộc, output, công cụ, trường hợp báo thiếu dữ liệu và lỗi.
3. Đánh giá điểm model bằng PR curve, một định nghĩa PR-AUC/AP ghi rõ, calibration plot và Brier score trên dữ liệu giữ riêng. Đây là phép đánh giá; nếu fit bộ hiệu chuẩn thì cần dùng train/validation hoặc cross-validation phù hợp, giữ TEST cho đánh giá cuối.
4. Bổ sung giải thích phù hợp với model đang dùng. Nếu dùng SHAP phải chỉ rõ model, dữ liệu nền, thang đầu ra và cách gom đặc trưng sau tiền xử lý; nếu dùng phương pháp khác phải ghi đúng tên.
5. Tách phần thông tin model, bằng chứng tài liệu và kết quả kiểm tra để người đọc biết chúng có vai trò gì.

### Những điều không nên sao chép nguyên

- Bài này tập trung một miền y khoa, không phải bộ thực nghiệm tổng hợp ML–CNN–chuỗi của bạn. Không cần đổi toàn bộ bố cục sang bảy chương giống nó.
- TF-IDF + cosine được gọi là “ngữ nghĩa” ở một số chỗ; dự án của bạn nên ghi đúng phương pháp thực thi.
- Đếm dấu [+]/[-] hoặc hai nhánh cùng đồng ý chưa chứng minh kết luận đúng; chúng có thể dùng chung yếu tố đầu vào và cùng sai.
- Không chuyển điểm của model BRFSS thành nhãn tiền tiểu đường, chẩn đoán hoặc ngưỡng điều trị tự đặt. BRFSS nhị phân của bạn không cung cấp nhãn phân biệt type 1/type 2.
- Fallback từ Neo4j sang NetworkX có thể giúp demo dễ chạy, nhưng nếu bổ sung phải cho biết backend thực sự đang dùng và có kiểm tra tương đương; không tuyên bố khả dụng 100% chỉ từ việc có fallback.

## 3. Bài Duy: kiểm tra độ bền và quản lý các lần cập nhật

### Những phần có thể học theo

- **Công khai ngân sách thực nghiệm.** Trang 6 phân biệt train pool và số mẫu thực sự cập nhật trọng số, thay vì chỉ ghi kích thước dataset gốc.
- **CNN trên ảnh và CNN trên bảng được phân biệt.** Trang 26–27 lý giải tích chập có cấu trúc không gian tự nhiên trên ảnh, trong khi thứ tự cột của bảng là một giả định cần kiểm tra.
- **Phép thử nhiễu đầu vào.** Trang 32 dùng các model truyền thống trên Digits, giữ model đã học trên ảnh sạch, thêm Gaussian noise vào TEST, lặp tạo nhiễu và báo trung bình/SD. Có thể học thiết kế phép thử để dùng cho CNN của bạn; không gán số liệu Digits của tác giả cho MNIST/CIFAR-10.
- **Đánh giá chuỗi qua nhiều giai đoạn.** Trang 38–39 có rolling-origin cho baseline và ghi rõ không xếp hạng chung với model sâu được thử ở một split khác. Đây là cách tránh so sánh sai giao thức.
- **Hỏi đáp về chính kết quả nghiên cứu.** Trang 40–44 dựng graph Report–Task–Evaluation, tra metric chính xác và tìm đoạn PDF có số trang. Bộ kiểm tra có 12 câu gắn trang, tách lookup khỏi retrieval và ghi rõ chưa chạy LLM sinh.
- **Bổ sung dữ liệu gốc được mô tả riêng.** Phụ lục B từ trang 93 trình bày thực nghiệm MNIST/CIFAR-10/Amazon/Avocado sau phần Digits/Sunspots ban đầu.

### Áp dụng vào dự án

1. Viết phép thử độ bền cho checkpoint ảnh hiện có: cùng mẫu test và cùng ảnh nhiễu cho các model; ghi seed, cách chuẩn hóa, biên độ nhiễu, số lần lặp và Accuracy/Macro-F1.
2. Với Amazon, trước tiên phân tích sai số theo các đoạn thời gian trong TEST hiện có. Nếu làm rolling-origin thì phải định nghĩa các fold và fit scaler/model theo dữ liệu hợp lệ của từng fold; không lấy một model đã thấy tương lai để dự đoán fold quá khứ.
3. Tạo bảng tra cứu kết quả: dataset, task, model, framework, split, metric, số mẫu, checkpoint hash và nguồn. Từ bảng này sinh số liệu cho báo cáo và metadata demo.
4. Có thể thêm công cụ hỏi “model nào tốt hơn trên tập test này?” bằng lookup từ registry, không yêu cầu LLM nhớ số liệu.
5. Báo cáo chỉ giữ một bản kết quả hiện hành cho mỗi giao thức; các lần sửa/cập nhật đặt ở lịch sử hoặc phụ lục và ghi phiên bản.

### Những điều cần tránh

Báo cáo Duy còn có lời ở phần cũ nói chưa có dữ liệu gốc trong khi phụ lục sau đã bổ sung. Khi tham khảo phải đọc cả trạng thái mới nhất. Bài bạn nên có một bảng trạng thái thống nhất, để phần chính, phụ lục, README và website không mô tả những hiện trạng khác nhau.

## 4. Bài Triều: thí nghiệm giải thích cơ chế và nối mô hình thành hệ thống

### Những phần có thể học theo

- **Cách đọc – nhận xét – giới hạn.** Trang 6 mô tả quy ước hộp giải thích dưới hình; trang 127 minh họa khá rõ bằng biểu đồ trí nhớ, loss và trạng thái cổng.
- **Thí nghiệm có câu hỏi cụ thể.** Trang 94–95 thử thay đổi thứ tự đặc trưng để đánh giá CNN trên bảng; trang 126–127 thử khả năng nhớ theo độ dài chuỗi và cách khởi tạo cổng. Những phép thử này giải thích cơ chế thay vì chỉ thêm model vào bảng.
- **Kiểm tra quy ước giữa framework.** Trang 128 trình bày cách lưu trọng số, bias và thứ tự cổng của RNN/LSTM/GRU. Đây là những thứ phải kiểm tra nếu chuyển checkpoint giữa Keras và PyTorch.
- **Kiểm chứng đường triển khai.** Trang 109–115 và 194 đối chiếu suy luận/đánh giá giữa môi trường Python và JavaScript. Ý tưởng đáng áp dụng là kiểm chứng đầu ra sau đóng gói; không nhất thiết chuyển frontend của bạn sang suy luận bằng JavaScript.
- **Đánh giá Agent bằng đối chứng cùng LLM.** Trang 205–206 phân biệt chỉ LLM, RAG cố định và Agent dùng công cụ; nhờ đó có thể tách ảnh hưởng của kiến trúc khỏi ảnh hưởng của việc đổi LLM. Có bộ câu hỏi cần nhiều bước riêng.
- **Mỗi khối kiến trúc gắn với module và dữ liệu thật.** Trang 213–214 nối các tác tử, công cụ và SQLite; thông điệp có thể truy vết. Trang 226–227 mô tả bộ tình huống và lỗi trích xuất, không chỉ đưa một điểm tổng.

### Áp dụng vào dự án

1. Dùng mẫu viết nhận xét: cách đọc hình → số liệu đáng chú ý → cách giải thích → giới hạn. Không cần lặp lý thuyết dài dưới mọi biểu đồ.
2. Có thể thêm một thí nghiệm trí nhớ tổng hợp nhỏ cho RNN/LSTM/GRU để minh họa tại sao cơ chế cổng có ích. Phải ghi rõ đây là dữ liệu sinh và kết quả này không thay thế khả năng dự báo Amazon.
3. Chuẩn hóa kiểm chứng checkpoint: nạp model, tái tạo preprocessing, so đầu ra với bộ mẫu tham chiếu và tính lại metric. Bạn đã có nhiều bước kiểm chứng này trong phiên làm việc; nên gom thành script tái lập ngay trong dự án.
4. Mỗi công cụ Agent có schema và lỗi rõ ràng. Nếu triển khai vòng lặp LLM cần giới hạn lượt, thời gian, kích thước kết quả và ghi những công cụ thực sự đã gọi.
5. SQLite có thể dùng cho run/metric/artifact/trace thử nghiệm để xem lại và so sánh. Chưa cần mở rộng sang hồ sơ bệnh nhân, OCR hay webhook ngoài phạm vi demo.
6. Chỉ cân nhắc ONNX/suy luận trình duyệt khi có nhu cầu chạy demo không có Python; đi kèm kiểm tra sai lệch. Flask hiện tại vẫn phù hợp cho bộ demo cục bộ.

### Những điều cần tránh

- Không cần sao chép toàn bộ phạm vi 239 trang, KGE, OCR, webhook, LLM API và nhiều Agent. Mỗi thành phần thêm vào phải có mục tiêu và đánh giá.
- Trang 232 nói rõ 55 ca được sinh từ 17 nguyên mẫu và đáp án do người viết quy tắc thiết kế. Điểm 100% là kiểm tra trên tập tổng hợp đó, không phải bằng chứng hiệu lực lâm sàng.
- Bộ nhớ đệm tái phát được kết quả đã ghi, không chứng minh một lần gọi LLM mới sẽ cho cùng kết quả.
- Nhiều câu hỏi Agent sinh từ graph đang dùng là một bộ kiểm tra chức năng có ích; cần thêm câu hỏi độc lập để đánh giá khả năng tổng quát.

## 5. Phần dự án đã có và phần nên bổ sung

| Hạng mục | Đã có trong bộ demo | Nâng cấp phù hợp |
|---|---|---|
| Trang demo | 24 trang, mỗi bài toán/kiến trúc một URL | Giữ cách tách trang; thêm metadata chạy và nguồn kết quả |
| Checkpoint và preprocessing | Có model, scaler/preprocessor, manifest hash | Gom kiểm chứng thành lệnh tái lập; ghi rõ split/config/threshold của từng model |
| Chỉ số đã lưu | Bảng ML/MLP/CNN/chuỗi và baseline | Sinh bảng/hình từ kết quả chuẩn; thêm PR/calibration, confusion matrix và phần dư khi có đủ dữ liệu |
| Độ tin cậy điểm dự đoán | Đã ghi chưa hiệu chuẩn | Đánh giá calibration; chỉ fit/calibrate/chọn threshold theo giao thức có validation hợp lệ |
| KG/truy xuất | Ba miền, KG/TF-IDF/Hybrid, lookup ngày, nguồn | Thêm loại câu hỏi thống kê, corpus có provenance và bộ câu hỏi độc lập; LLM là một giai đoạn bổ sung riêng |
| Agent | Điều phối theo luật, kiểm tra input, gọi MLP và truy xuất, trace | Schema trạng thái/tool; kiểm tra output; đánh giá đúng/sai/thiếu thông tin/lỗi công cụ |
| Kiểm thử | 16 nhóm test; đã có kiểm tra API và nhiều model | Formalize kiểm chứng đầu ra tham chiếu, artifact integrity, đo thời gian và thử lỗi công cụ |
| Báo cáo | Đề cương sáu chương và quy ước đã chốt | Thêm phần thiết kế, bảng giao thức, phân tích lỗi và phụ lục tái lập |

## 6. Danh mục nâng cấp theo mức ưu tiên

Đây là thứ tự đề xuất để triển khai tiếp, chưa phải kết quả đã hoàn tất.

| Ưu tiên | Hạng mục | Vị trí code dự kiến | Điều kiện hoàn thành |
|---|---|---|---|
| P0 | Registry thí nghiệm và nguồn số liệu thống nhất | `experiments/registry.json`, `scripts/build_report_assets.py` | Hình/bảng truy được đến prediction/metric, dataset, split và checkpoint; dữ liệu cũ được ghi thiếu thông tin thay vì điền giả |
| P0 | Script kiểm chứng artifact và đầu ra tham chiếu | `scripts/verify_artifacts.py`, `tests/fixtures/` | Hash đúng; preprocessing/model đúng; output khớp dung sai đã định; metric tái tạo từ TEST |
| P0 | Ba sơ đồ thiết kế theo hệ thống thật | `report/figures/`, tài liệu schema | Kiến trúc tổng thể, sequence và cấu trúc dữ liệu đúng với API/module; phần đề xuất ghi rõ |
| P1 | Phân tích phân loại đầy đủ | `experiments/ch02_tabular/evaluate.py` | Confusion matrix, Precision/Recall/F1, PR/AP và calibration từ prediction giữ riêng; lựa chọn ngưỡng không dùng TEST |
| P1 | Độ bền CNN | `experiments/ch03_cnn/robustness.py` | Cùng test cases/nhiễu cho các model; seed và điều kiện được lưu; clean/noisy được báo riêng |
| P1 | Sai số chuỗi theo giai đoạn và baseline | `experiments/ch04_sequences/evaluate.py` | Tất cả model cùng target/time index; MAE/RMSE theo đoạn; không biến dự báo một bước thành nhiều bước bằng đổi nhãn |
| P1 | Benchmark Agent theo nhóm nhiệm vụ | `experiments/ch06_agents_rl/evaluate_agents.py` | Định nghĩa pass/fail trước; có input thiếu/sai, không có nguồn và lỗi tool; thành công điều phối tách độ đúng model |
| P2 | RAG dense và sinh có nguồn | `demo/knowledge.py`, `experiments/ch05_kg_rag/` | Model thực sự chạy; corpus/DEV/TEST và log prompt/context/output rõ; đo retrieval và generation riêng |
| P2 | RL và thí nghiệm trí nhớ | `experiments/ch06_agents_rl/`, `experiments/ch04_sequences/` | Dữ liệu mô phỏng được ghi rõ; train/eval tách; baseline và nhiều seed khi có điều kiện |
| P2 | Agent dùng LLM, SQLite, ONNX | Module tương ứng sau khi có nhu cầu | Chứng minh lợi ích cụ thể và đánh giá chi phí, lỗi, độ tương đương; không đổi công nghệ chỉ để giống bài mẫu |

### Nâng cấp đầu ra API được đề xuất

Kết quả hiện đã có timestamp và latency. Có thể bổ sung metadata dùng cho chế độ xem kỹ thuật/xuất JSON:

```json
{
  "run_id": "...",
  "task": "stock",
  "model_id": "AMZN_Keras_GRU",
  "artifact_sha256": "...",
  "preprocessing_version": "...",
  "method": "inference",
  "status": "ok",
  "result": {},
  "sources": [],
  "tools": [],
  "latency_ms": 0,
  "timestamp": "..."
}
```

Đây là schema minh họa, không phải response đã được thay đổi. Phần giao diện chính chỉ cần thông tin giúp người dùng hiểu kết quả; metadata đầy đủ nằm trong phần kỹ thuật/xuất JSON.

## 7. Bổ sung vào báo cáo theo khung sáu chương

- **Mở đầu:** câu hỏi nghiên cứu, bảng dữ liệu/nhiệm vụ/model/sản phẩm và sơ đồ liên kết các chương.
- **Chương 2:** data dictionary, phân bố nhãn, giao thức chia tập, threshold, PR và calibration; phân tích sai theo nhóm phù hợp. BRFSS Age là mã nhóm tuổi.
- **Chương 3:** shape và số tham số theo từng khối, ảnh lỗi, confusion matrix; thêm một phép thử độ bền có kiểm soát.
- **Chương 4:** sơ đồ tạo cửa sổ, mốc thời gian train/val/test, baseline và sai số theo giai đoạn; thí nghiệm trí nhớ là mục mở rộng tùy chọn.
- **Chương 5:** schema graph, nguồn của node/cạnh/đoạn, bảng loại câu hỏi và công cụ trả lời, evaluation với câu ngoài miền và câu diễn đạt lại.
- **Chương 6:** bảng Agent/tool/input/output; sequence, trace, trạng thái thành công/thiếu dữ liệu/lỗi; RL là thực nghiệm riêng có môi trường và reward.
- **Phụ lục:** môi trường, lệnh chạy, cấu hình, danh mục artifact, ánh xạ figure/table → script → kết quả và test report.

Một hình/đoạn code nên đi cùng phần giải thích: đầu vào/đầu ra, bước quan trọng, kết quả quan sát được và giới hạn. Không cần giải thích từng dòng import hoặc biến vòng lặp hiển nhiên.

## 8. Cách chọn lọc phù hợp nhất

Giữ bố cục thực nghiệm đã chọn; học thiết kế hệ thống và phân tích điểm dự đoán từ Việt, sự rõ ràng về pipeline và luồng KG/CSV/RAG từ Khôi, kiểm tra độ bền và giao thức từ Duy, cách giải thích cơ chế và kiểm chứng triển khai từ Triều.

Ưu tiên đầu tiên là làm số liệu, sơ đồ, code và checkpoint khớp nhau. Sau đó tăng chiều sâu bằng một số thí nghiệm có câu hỏi rõ ràng. Những phần còn thiếu đã ghi trong đề cương như RAG sinh và hai demo RL vẫn cần làm nếu theo phạm vi đã chọn; việc đọc thêm báo cáo không biến chúng thành chức năng đã có.
