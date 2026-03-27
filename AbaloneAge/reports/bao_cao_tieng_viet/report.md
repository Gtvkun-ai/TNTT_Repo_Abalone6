# Báo cáo dự án AbaloneAge

## 1. Giới thiệu đề tài

Bài toán của dự án là dự đoán số vòng tuổi của bào ngư dựa trên các đặc trưng hình thái và khối lượng cơ thể. Trong thực tế, tuổi của bào ngư thường được xác định bằng cách cắt vỏ, nhuộm và đếm số vòng dưới kính hiển vi. Cách làm này tốn thời gian và khó áp dụng trên quy mô lớn, vì vậy việc sử dụng các mô hình học máy để ước lượng tuổi từ các phép đo dễ thu thập hơn có ý nghĩa thực tiễn rõ rệt.

Trong bộ dữ liệu Abalone, biến mục tiêu là `Rings`. Theo tài liệu gốc, tuổi gần đúng có thể được suy ra bằng công thức:

`Age = Rings + 1.5`

Tuy nhiên, trong phạm vi dự án này, nhóm lựa chọn dự đoán trực tiếp `Rings` để bám sát dữ liệu gốc và thuận tiện cho việc xây dựng các mô hình hồi quy bằng scikit-learn.

Mục tiêu chính của dự án gồm:

- hiểu cấu trúc và đặc điểm của bộ dữ liệu Abalone
- xây dựng pipeline xử lý dữ liệu và huấn luyện mô hình
- thử nghiệm đủ 10 thuật toán hồi quy bắt buộc theo đề bài
- đánh giá mô hình bằng các thang đo phù hợp
- thực hiện tối ưu hóa và so sánh giữa baseline và optimized model

## 2. Mô tả dữ liệu

Theo file `data/raw/abalone.names`, bộ dữ liệu Abalone có:

- `4177` mẫu
- `8` thuộc tính đầu vào
- `1` biến mục tiêu là `Rings`

Trong dự án, các cột được sử dụng gồm:

- `Sex`
- `Length`
- `Diameter`
- `Height`
- `WholeWeight`
- `ShuckedWeight`
- `VisceraWeight`
- `ShellWeight`
- `Rings`

Ý nghĩa các thuộc tính:

- `Sex`: giới tính của bào ngư, gồm `M`, `F`, `I`
- `Length`: chiều dài lớn nhất của vỏ
- `Diameter`: đường kính vuông góc với chiều dài
- `Height`: chiều cao
- `WholeWeight`: khối lượng toàn bộ
- `ShuckedWeight`: khối lượng phần thịt
- `VisceraWeight`: khối lượng nội tạng
- `ShellWeight`: khối lượng vỏ sau khi sấy
- `Rings`: số vòng trên vỏ, được dùng làm biến đích

Dựa trên dữ liệu gốc và notebook khám phá dữ liệu, có thể rút ra một số đặc điểm chính:

- dữ liệu không có giá trị thiếu
- có một biến phân loại là `Sex`
- các biến còn lại chủ yếu là biến số liên tục
- một số biến khối lượng và cả `Rings` có độ lệch phân phối và xuất hiện ngoại lệ
- quan hệ giữa các biến đầu vào với `Rings` có thể không hoàn toàn tuyến tính

Những quan sát này là cơ sở cho các bước tiền xử lý và lựa chọn mô hình ở các phần sau.

## 3. Quy trình thực hiện

Quy trình thực hiện trong repo được xây dựng bám theo các notebook từ `01` đến `05`, đồng thời phù hợp với yêu cầu của đề bài.

### 3.1. Load Data

Dữ liệu được đọc từ file `data/raw/abalone.csv` thông qua module `src/data/load_data.py`. Trong bước này, nhóm gán lại tên cột chuẩn cho toàn bộ dataset để thuận tiện cho các bước phân tích và xử lý tiếp theo.

### 3.2. Initial Data Analysis

Ở notebook `01_kham_pha_du_lieu.ipynb`, nhóm thực hiện kiểm tra ban đầu gồm:

- kích thước dữ liệu
- kiểu dữ liệu từng cột
- thống kê mô tả
- missing values
- duplicate records

Kết quả cho thấy dữ liệu có `4177` dòng, không có missing value, phù hợp để tiếp tục triển khai bài toán hồi quy mà không cần bước điền khuyết dữ liệu.

### 3.3. Exploratory Data Analysis

Sau phân tích ban đầu, nhóm thực hiện EDA để hiểu rõ hơn bản chất dữ liệu:

- phân tích phân phối biến mục tiêu `Rings`
- phân tích biến phân loại `Sex`
- trực quan hóa các biến số
- kiểm tra outlier
- xem xét tương quan giữa các biến và `Rings`

Từ EDA, nhóm rút ra các kết luận chính:

- `Sex` có liên hệ nhất định với biến mục tiêu nên cần được giữ lại
- dữ liệu có ngoại lệ và độ lệch ở một số biến khối lượng
- các mô hình phi tuyến có thể phù hợp vì quan hệ giữa đầu vào và `Rings` không hoàn toàn tuyến tính

### 3.4. Preprocess Data

Notebook `02_tien_xu_ly_du_lieu.ipynb` thực hiện các bước tiền xử lý dựa trên kết quả EDA:

- làm sạch dữ liệu cơ bản
- chuẩn hóa tên cột
- mã hóa biến `Sex` bằng One-Hot Encoding
- chia dữ liệu theo tỷ lệ `70% train`, `30% test`
- cố định `random_state = 42`
- chuẩn bị nhiều phiên bản dữ liệu cho các nhóm mô hình khác nhau

Ba phiên bản dữ liệu chính được tạo ra là:

- `encoded_only`
- `standard_scaled`
- `robust_log_scaled`

Trong đó:
- `encoded_only` phù hợp với mô hình cây
- `standard_scaled` phù hợp với các mô hình nhạy với scale
- `robust_log_scaled` dùng để kiểm tra ảnh hưởng của outlier và độ lệch phân phối

### 3.5. Build Models

Notebook `03_huan_luyen_mo_hinh.ipynb` triển khai đủ 10 mô hình hồi quy bắt buộc:

1. `LinearRegression`
2. `KNeighborsRegressor`
3. `RidgeRegression`
4. `DecisionTreeRegressor`
5. `RandomForestRegressor`
6. `GradientBoostingRegressor`
7. `SGDRegressor`
8. `SVR`
9. `LinearSVR`
10. `MLPRegressor`

Các mô hình được ghép với phiên bản dữ liệu phù hợp thay vì dùng một kiểu preprocess duy nhất cho tất cả. Đây là một lựa chọn hợp lý vì mô hình cây và mô hình nhạy scale có yêu cầu đầu vào khác nhau.

### 3.6. Evaluate Models

Việc đánh giá trong quá trình huấn luyện được thực hiện bằng `5-fold Cross-validation`, đúng yêu cầu đề bài. Các metric sử dụng trong dự án gồm:

- `MSE`
- `RMSE`
- `RSE`
- `MAE`
- `Execution Time`

Notebook `03_huan_luyen_mo_hinh.ipynb` dùng dự đoán OOF từ cross-validation để so sánh baseline giữa các mô hình.

### 3.7. Assess Model

Notebook `04_danh_gia_mo_hinh.ipynb` được dùng để đánh giá cuối cùng trên tập test. Ở bước này, nhóm:

- so sánh kết quả CV với kết quả test
- chọn các mô hình tốt nhất theo metric
- vẽ biểu đồ so sánh giữa giá trị thực và giá trị dự đoán
- phân tích residual plot

Đây là bước quan trọng để đánh giá khả năng tổng quát hóa thật sự của từng mô hình.

### 3.8. Tối ưu hóa mô hình

Notebook `05_thu_nghiem_bo_sung.ipynb` triển khai phần cải thiện và tối ưu hóa, bao gồm ít nhất 5 hướng:

1. So sánh `standard_scaled` và `robust_log_scaled`
2. Tuning toàn bộ 10 mô hình bằng `RandomizedSearchCV`
3. Chọn phiên bản dữ liệu tốt hơn cho từng mô hình trước khi tuning
4. Xây dựng `Weighted Average Ensemble`
5. Xây dựng `Stacking` với meta-model `Ridge`

Phần này giúp nhóm so sánh rõ giữa baseline và optimized model theo đúng tinh thần đề bài.

## 4. Kết quả

Dựa trên cấu trúc notebook và phần tổng kết trong notebook `05_thu_nghiem_bo_sung.ipynb`, repo hiện đã có hướng so sánh giữa baseline, optimized và ensemble.

Các kết quả chính được ghi nhận trong notebook gồm:

- mô hình baseline tốt nhất trên tập test là `MLPRegressor`
- `RMSE_test` của baseline `MLPRegressor` xấp xỉ `2.097`
- mô hình optimized tốt nhất hiện tại là `SVR`
- `RMSE_test` của `SVR` sau tối ưu xấp xỉ `2.142`
- các ensemble như `Weighted Average Ensemble` và `StackingRidge` đạt `RMSE_test` khoảng `2.11 - 2.13`

Từ đó có thể rút ra một số nhận xét:

- việc tối ưu hóa giúp cải thiện một số kết quả trong cross-validation
- tuy nhiên trên tập test hiện tại, baseline `MLPRegressor` vẫn cho kết quả tốt nhất
- điều này cho thấy tối ưu hóa không phải lúc nào cũng giúp mô hình tổng quát hóa tốt hơn trên dữ liệu chưa thấy
- các mô hình ensemble là hướng mở rộng hợp lý nhưng trong kết quả hiện tại vẫn chưa vượt được baseline mạnh nhất

Ngoài ra, dự án đã chuẩn bị các dạng đánh giá trực quan như:

- biểu đồ so sánh metric giữa các mô hình
- biểu đồ `Actual vs Predicted`
- residual plot

Đây là các thành phần hữu ích để đưa trực tiếp vào báo cáo và slide khi hoàn thiện bản nộp cuối.

## 5. Kết luận

Dự án `AbaloneAge` đã được tổ chức theo một pipeline tương đối đầy đủ cho bài toán hồi quy dự đoán `Rings` của bào ngư. Nhóm đã:

- phân tích dữ liệu ban đầu và thực hiện EDA
- xây dựng các bước tiền xử lý dựa trên kết quả khám phá dữ liệu
- thử đủ 10 mô hình hồi quy bắt buộc
- đánh giá bằng `5-fold Cross-validation` và trên tập test
- triển khai phần tối ưu hóa và thử thêm các chiến lược ensemble

Kết quả đáng chú ý nhất hiện tại là baseline `MLPRegressor` vẫn cho hiệu năng tốt nhất trên tập test, trong khi các mô hình tối ưu hóa và ensemble chưa vượt qua được kết quả đó. Về mặt kỹ thuật, đây là một kết luận có giá trị vì nó phản ánh đúng sự khác biệt giữa cải thiện trên tập huấn luyện và khả năng tổng quát hóa trên dữ liệu mới.

Trong thời gian tới, nhóm có thể tiếp tục hoàn thiện dự án bằng cách:

- chạy lại đầy đủ notebook để sinh file metric thật vào `outputs/metrics`
- bổ sung bảng kết quả chi tiết cho cả 10 mô hình
- thêm hình minh họa từ notebook vào báo cáo
- mở rộng feature engineering và tuning sâu hơn cho các mô hình mạnh
