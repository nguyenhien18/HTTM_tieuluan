"""Independent demos: one task and one model family per URL."""
GROUPS=[dict(id='ml',title='Machine Learning',description='Mô hình truyền thống trên dữ liệu bảng.'),dict(id='mlp',title='Mạng nơ-ron MLP',description='Mạng MLP huấn luyện theo quy trình có kiểm chứng.'),dict(id='cnn',title='CNN · Thị giác',description='Một bộ dữ liệu ảnh trên mỗi trang.'),dict(id='rnn',title='RNN · Dữ liệu chuỗi',description='Hành vi khách hàng và giá cổ phiếu.'),dict(id='knowledge',title='Tri thức & Agent',description='Đồ thị, hỏi đáp có nguồn và phối hợp tác vụ.'),dict(id='a01',title='Demo A01',description='Hai bài toán nhập môn của A01.')]
def page(id,path,group,title,description,kind,icon,**config):
    return dict(id=id,path=path,group=group,title=title,description=description,kind=kind,icon=icon,**config)
PAGES=[]
for engine,group,assignment in [('ml','ml','A02'),('mlp','mlp','A03')]:
    for task,title,description in [('diabetes','Tiểu đường BRFSS','Phân loại từ 21 đặc trưng khảo sát sức khỏe.'),('house','Giá nhà Ba Lan','Dự đoán theo thông tin căn nhà và vị trí.'),('ecommerce','Khả năng mua hàng','Dự đoán từ 16 đặc trưng hành vi khách hàng.')]:
        PAGES.append(page(f'{engine}_{task}',f'/{engine}/{task}',group,f'{engine.upper()} · {title}',description,'tabular','ml',task=task,engine=engine,badge=assignment,kicker=('MACHINE LEARNING' if engine=='ml' else 'NEURAL NETWORK')+' / '+title.upper()))
for dataset,slug,title,description,assignment in [('mnist','mnist','Chữ số viết tay','Chọn ảnh MNIST, vẽ một chữ số hoặc tải ảnh để nhận diện.','A04'),('fashion','fashion-mnist','Phân loại trang phục','Thử các kiến trúc CNN trên ảnh Fashion-MNIST.','A05'),('cifar10','cifar10','Nhận diện ảnh màu','Phân loại ảnh CIFAR-10 bằng các kiến trúc CNN.','A05')]:
    PAGES.append(page('cnn_'+dataset,'/cnn/'+slug,'cnn','CNN · '+title,description,'cnn','cnn',dataset=dataset,badge=assignment,kicker='CONVOLUTIONAL NETWORK / '+slug.upper()))
for task,slug,title,description in [('customer','customer','Hành vi khách hàng','Dùng 5 hành vi gần nhất để dự đoán sự kiện tiếp theo.'),('stock','amazon','Giá Amazon','Dự đoán một phiên lịch sử và đối chiếu với giá thật, baseline.')]:
    PAGES.append(page('rnn_'+task,'/rnn/'+slug,'rnn','RNN · '+title,description,'rnn','rnn',task=task,badge='A06 · SIMPLE RNN',kicker='RECURRENT NETWORK / '+slug.upper()))
PAGES += [page('knowledge_graph','/knowledge/graph','knowledge','Knowledge Graph','Khám phá thực thể, quan hệ và tài liệu tham khảo trong đồ thị.','graph','knowledge',badge='ĐỒ THỊ CÓ NGUỒN',kicker='KNOWLEDGE / GRAPH'),page('knowledge_retrieval','/knowledge/retrieval','knowledge','Hỏi đáp có nguồn','Đặt câu hỏi và đối chiếu bằng chứng; tùy chọn sinh bằng Qwen cục bộ.','retrieval','knowledge',badge='TF-IDF · QWEN CỤC BỘ',kicker='KNOWLEDGE / RETRIEVAL'),page('agents_diabetes','/agents/diabetes','knowledge','Agent · Công cụ và kiểm định','Định tuyến ba miền, gọi công cụ, kiểm tra đầu ra và ghi trace.','agents','agents',badge='ROUTER · VERIFIER · RETRY',kicker='CHƯƠNG 6 / TOOL AGENTS'),page('a01_diabetes','/a01/diabetes','a01','A01 · Tiểu đường','Demo KNN từ tuổi, BMI và đường huyết.','tabular','ml',task='diabetes_a01',engine='ml',badge='A01 · KNN',kicker='ASSIGNMENT 01 / DIABETES'),page('a01_house','/a01/house','a01','A01 · Giá nhà Việt Nam','Dự đoán giá nhà theo đặc điểm vật lý và tỉnh, thành phố.','tabular','ml',task='house_a01',engine='ml',badge='A01 · RANDOM FOREST',kicker='ASSIGNMENT 01 / GIÁ NHÀ')]

GROUPS.insert(4,dict(id='gated',title='LSTM & GRU · Chương 4',description='Các checkpoint Keras đã lưu của A06_RNN.'))
GROUPS.insert(5,dict(id='domains',title='KG & Truy xuất · Chương 5',description='Ba miền độc lập; đối chiếu KG, BM25, TF-IDF và kết hợp.'))
GROUPS.append(dict(id='report',title='Thực nghiệm & báo cáo',description='Kiểm chứng pipeline và đối chiếu các chương.'))
for architecture in ['lstm','gru']:
    for task,slug,title in [('customer','customer','Hành vi khách hàng'),('stock','amazon','Giá Amazon')]:
        PAGES.append(page(architecture+'_'+task,'/'+architecture+'/'+slug,'gated',architecture.upper()+' · '+title,'Chạy checkpoint Keras và đối chiếu cùng tập test A06.','rnn','rnn',task=task,architecture=architecture,badge='KERAS · CHECKPOINT THẬT',kicker='CHƯƠNG 4 / '+architecture.upper()))
for domain,title in [('diabetes','Tiểu đường'),('stock','Cổ phiếu Amazon'),('gold','Giá vàng')]:
    PAGES.append(page('domain_'+domain,'/knowledge/'+domain,'domains','KG · '+title,'Xem bản ghi, quan hệ và bằng chứng theo từng phương pháp truy xuất.','domain','knowledge',domain=domain,badge='KG · BM25 · TF-IDF · HYBRID',kicker='CHƯƠNG 5 / '+title.upper()))
PAGES.append(page('experiments','/experiments','report','Kết quả thực nghiệm','Đối chiếu dữ liệu, split, checkpoint, đánh giá và các trường hợp lỗi.','experiments','agents',badge='KẾT QUẢ CÓ TRUY VẾT',kicker='EXPERIMENTS / CHƯƠNG 2–6'))
GROUPS.insert(-1,dict(id='rl',title='Học tăng cường · Chương 6',description='Huấn luyện Q-learning trong môi trường mô phỏng có đối chứng.'))
PAGES.extend([
 page('rl_maze','/rl/maze','rl','Q-learning · Mê cung','Huấn luyện Agent, xem đường đi và so với random/BFS.','rl','agents',task='maze',badge='MÔ PHỎNG · Q-TABLE',kicker='CHƯƠNG 6 / MAZE'),
 page('rl_pricing','/rl/pricing','rl','Q-learning · Cạnh tranh giá','So sánh chính sách học được, agent luật và mốc biết mô hình.','rl','agents',task='pricing',badge='MÔ PHỎNG · CÙNG ĐIỀU KIỆN',kicker='CHƯƠNG 6 / PRICING')
])
