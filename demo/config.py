from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets'
DATA = ASSETS / 'data'
MODELS = ASSETS / 'models'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def number(name, label, low, high, default, step=1, hint=''):
    return dict(name=name, label=label, kind='number', min=low, max=high, default=default, step=step, hint=hint)

def choice(name, label, options, default=None):
    pairs=[{'value':v,'label':str(t)} for v,t in options] if options and isinstance(options[0],tuple) else [{'value':v,'label':str(v)} for v in options]
    return dict(name=name,label=label,kind='select',options=pairs,default=default if default is not None else pairs[0]['value'])

BINARY = [('HighBP','Huyết áp cao'),('HighChol','Cholesterol cao'),('CholCheck','Đã kiểm tra cholesterol'),('Smoker','Từng hút ≥100 điếu thuốc'),('Stroke','Tiền sử đột quỵ'),('HeartDiseaseorAttack','Bệnh tim / nhồi máu'),('PhysActivity','Có vận động thể chất'),('Fruits','Ăn trái cây hằng ngày'),('Veggies','Ăn rau hằng ngày'),('HvyAlcoholConsump','Uống nhiều rượu bia'),('AnyHealthcare','Có bảo hiểm / dịch vụ y tế'),('NoDocbcCost','Khó khám vì chi phí'),('DiffWalk','Khó đi bộ / leo cầu thang')]
DIABETES = [number('BMI','BMI',10,100,27,.1),choice('Age','Nhóm tuổi',[(i, f'{18 if i==1 else 25+(i-2)*5}–{24 if i==1 else 29+(i-2)*5} tuổi' if i<13 else '80 tuổi trở lên') for i in range(1,14)],9),choice('Sex','Giới tính',[(0,'Nữ'),(1,'Nam')],1),choice('GenHlth','Sức khỏe tự đánh giá',[(1,'Rất tốt'),(2,'Tốt'),(3,'Khá'),(4,'Trung bình'),(5,'Kém')],3),number('MentHlth','Số ngày sức khỏe tinh thần kém',0,30,0),number('PhysHlth','Số ngày sức khỏe thể chất kém',0,30,0),choice('Education','Trình độ học vấn',[(1,'Chưa đi học'),(2,'Tiểu học'),(3,'Chưa xong THPT'),(4,'Tốt nghiệp THPT'),(5,'Một phần cao đẳng / đại học'),(6,'Tốt nghiệp cao đẳng / đại học')],5),choice('Income','Nhóm thu nhập năm (USD)',[(1,'< 10.000'),(2,'10.000–14.999'),(3,'15.000–19.999'),(4,'20.000–24.999'),(5,'25.000–34.999'),(6,'35.000–49.999'),(7,'50.000–74.999'),(8,'≥ 75.000')],6)] + [choice(k,label,[(0,'Không'),(1,'Có')],1 if k in ['CholCheck','PhysActivity','Fruits','Veggies','AnyHealthcare'] else 0) for k,label in BINARY]
profiles=read_json(DATA/'city_profiles.json')
HOUSE=[choice('city_name','Thành phố',sorted(profiles),'Warszawa'),number('area','Diện tích (m²)',10,500,70,.1),number('rooms','Số phòng',1,20,3),number('floor','Tầng',-1,20,2),choice('offer_type','Người đăng',['Private','Business'],'Private'),choice('offer_type_of_building','Loại nhà',['Housing Block','Apartment Building','Tenement','Detached House','Other'],'Housing Block'),choice('market','Thị trường',['aftermarket','primary'],'aftermarket'),choice('month','Tháng đăng',['January','February','March'],'March')]
house_categories=read_json(DATA/'house_categories.json')
for field in HOUSE:
    if field['name'] in house_categories:
        field['options']=[dict(value=v,label=v) for v in house_categories[field['name']]]
        if field['default'] not in house_categories[field['name']]:field['default']=house_categories[field['name']][0]
ECOMMERCE=[number('Age','Tuổi',15,81,35),number('AnnualIncome','Thu nhập năm',10000,250000,65000,100),number('NumberOfPurchases','Số lần mua trước đây',0,50,8),number('TimeSpentOnWebsite','Thời gian trên website (phút)',0,100,24,.1),number('CustomerTenureYears','Số năm là khách hàng',0,20,3,.1),number('LastPurchaseDaysAgo','Số ngày từ lần mua gần nhất',0,250,18),number('DiscountsAvailed','Số lần dùng ưu đãi',0,10,2),number('SessionCount','Số phiên truy cập',1,20,5),number('CustomerSatisfaction','Mức hài lòng',1,5,4),choice('Gender','Giới tính',['Female','Male'],'Female'),choice('ProductCategory','Nhóm sản phẩm',['Electronics','Fashion','Furniture','Groceries','Kitchen'],'Electronics'),choice('PreferredDevice','Thiết bị',['Desktop','Mobile','Tablet'],'Mobile'),choice('Region','Khu vực',['East','North','South','West'],'South'),choice('ReferralSource','Nguồn truy cập',['Email','Organic','Paid Ads','Referral','Social'],'Organic'),choice('CustomerSegment','Phân khúc',['Premium','Regular','VIP'],'Regular'),choice('LoyaltyProgram','Khách hàng thân thiết',[(0,'Không'),(1,'Có')],1)]
CITIES_VN=read_json(MODELS/'a01/cities.json')
TASKS={
 'diabetes':dict(title='Tiểu đường · BRFSS',source='A02 + A03',subtitle='21 đặc trưng sức khỏe và lối sống',fields=DIABETES,engines=['ml','mlp','compare'],kind='classification',note='Minh họa phân loại dữ liệu khảo sát. Điểm mô hình chưa được hiệu chỉnh thành xác suất chẩn đoán.'),
 'house':dict(title='Giá nhà · Ba Lan',source='A02 + A03',subtitle='Đặc điểm căn nhà và vị trí',fields=HOUSE,engines=['ml','mlp','compare'],kind='regression',unit='PLN',note='Dự đoán theo dữ liệu tin đăng Ba Lan, không phải giá nhà Việt Nam. Tên thành phố được chuẩn hóa nhất quán.'),
 'ecommerce':dict(title='Khả năng mua hàng',source='A02 + A03',subtitle='16 đặc trưng hành vi khách hàng',fields=ECOMMERCE,engines=['ml','mlp','compare'],kind='classification',note='Kết quả minh họa trên bộ dữ liệu của bài thực hành; không suy ra hiệu quả kinh doanh ngoài bộ dữ liệu.'),
 'diabetes_a01':dict(title='Tiểu đường · A01',source='A01',subtitle='Tuổi, BMI và đường huyết',fields=[number('age','Tuổi',1,80,50),number('bmi','BMI',10,96,29,.1),number('blood_glucose_level','Đường huyết (mg/dL)',80,300,160)],engines=['ml'],kind='classification',note='KNN trên 3 đặc trưng. Điểm hiển thị là tỷ lệ phiếu cho lớp dương, không phải độ chắc chắn của chẩn đoán.'),
 'house_a01':dict(title='Giá nhà · Việt Nam',source='A01',subtitle='Đặc điểm căn nhà và tỉnh / thành phố',fields=[choice('City','Tỉnh / thành phố',CITIES_VN,'Hà Nội'),number('Area','Diện tích (m²)',3.1,595,70,.1),number('Frontage','Mặt tiền (m)',1,77,5,.1),number('Access Road','Đường vào (m)',1,85,5,.1),number('Floors','Số tầng',1,10,3),number('Bedrooms','Phòng ngủ',1,9,3),number('Bathrooms','Phòng tắm',1,9,3)],engines=['ml'],kind='regression',unit='tỷ VNĐ',note='Dự đoán theo dữ liệu tin đăng của A01, đơn vị tỷ VNĐ.')
}
CNN={
 'mnist':dict(title='Chữ số viết tay',description='MNIST · ảnh xám 28 × 28',classes=[str(i) for i in range(10)],models=[('torch_improved','PyTorch · Improved'),('torch_basic','PyTorch · Basic'),('numpy_scratch','NumPy · Scratch'),('keras_improved','Keras · Improved'),('keras_basic','Keras · Basic')]),
 'fashion':dict(title='Phân loại trang phục',description='Fashion-MNIST · ảnh xám 28 × 28',classes=['Áo thun','Quần dài','Áo chui đầu','Váy','Áo khoác','Dép sandal','Áo sơ mi','Giày sneaker','Túi','Bốt cổ thấp'],models=[('basic_cnn','Basic CNN'),('lenet_like','LeNet-like'),('vgg_like','VGG-like'),('resnet_like','ResNet-like')]),
 'cifar10':dict(title='Nhận diện ảnh màu',description='CIFAR-10 · ảnh màu 32 × 32',classes=['Máy bay','Ô tô','Chim','Mèo','Hươu','Chó','Ếch','Ngựa','Tàu thủy','Xe tải'],models=[('resnet_like','ResNet-like'),('basic_cnn','Basic CNN'),('lenet_like','LeNet-like'),('vgg_like','VGG-like')])
}
