# BẢN ĐẶC TẢ KỸ THUẬT CHI TIẾT: ĐỒ ÁN HỌC MÁY (SCIKIT-LEARN)

## 1. Mục tiêu và Phạm vi Tiếp cận

Đồ án yêu cầu người thực hiện phải chứng minh năng lực vận dụng thư viện scikit-learn để xử lý các bài toán thực tế. Hai yếu tố sống còn quyết định điểm số bao gồm:

- Đánh giá Hiệu suất (Performance Metrics): Sử dụng các thang đo định lượng để phản ánh chính xác trạng thái mô hình.
- Tối ưu hóa (Optimization): Áp dụng các kỹ thuật nâng cao để cải thiện kết quả so với mô hình cơ sở.

## 2. Danh mục Dữ liệu (15 Bộ dữ liệu lựa chọn)

### 2.1. Phân loại Nhị phân (Binary Classification) - 07 Bộ

- Bank Note Authentication: Xác thực tiền giấy.
- Horse Colic: Chẩn đoán đau bụng ở ngựa.
- Sonar: Phân biệt đá và mỏ hình trụ.
- German Credit: Đánh giá tín dụng Đức.
- Ionosphere: Cấu trúc tầng điện ly.
- Oil Spill: Phát hiện vết dầu loang.
- Phoneme: Phân loại âm vị.

### 2.2. Phân loại Đa lớp (Multiclass Classification) - 04 Bộ

- Thyroid Gland: Chẩn đoán bệnh tuyến giáp.
- Ecoli: Phân loại protein vi khuẩn E.coli.
- Glass Identification: Phân loại các loại thủy tinh.
- Wheat Seeds: Phân loại hạt giống lúa mì.

### 2.3. Hồi quy (Regression) - 04 Bộ

- Abalone Age: Dự đoán tuổi bào ngư.
- Wine Quality Red: Dự đoán chất lượng rượu vang đỏ.
- Auto Insurance: Dự đoán mức bồi thường bảo hiểm ô tô.
- Auto Imports Prices: Dự đoán giá xe ô tô nhập khẩu.

## 3. Quy trình Triển khai Kỹ thuật (07 Giai đoạn)

Không được phép bỏ qua bất kỳ bước nào trong chuỗi logic sau:

1. Giai đoạn 1 - Load Data: Tải và nạp dữ liệu vào môi trường làm việc.
2. Giai đoạn 2 - Initial Data Analysis: Phân tích sơ bộ về kích thước, kiểu dữ liệu và phân phối cơ bản.
3. Giai đoạn 3 - Exploratory Data Analysis (EDA): Phân tích khám phá chuyên sâu. Sử dụng biểu đồ để tìm ra mối tương quan, giá trị ngoại lai (outliers) và đặc trưng quan trọng.
4. Giai đoạn 4 - Preprocess Data: Tiền xử lý dữ liệu.

- Loại bỏ nhiễu, xử lý giá trị thiếu.
- Trích xuất hoặc lựa chọn đặc trưng (Feature Engineering).
- Yêu cầu: Mọi hành động tại bước này phải có bằng chứng đối chiếu từ kết quả EDA.

5. Giai đoạn 5 - Build Models: Xây dựng mô hình trên tập huấn luyện (Training data).
6. Giai đoạn 6 - Evaluate Models: Đánh giá độ ổn định của mô hình bằng kiểm tra chéo (k-fold Cross-validation).
7. Giai đoạn 7 - Assess Model: Đánh giá năng lực tổng quát hóa cuối cùng trên tập kiểm tra (Test data).

## 4. Cấu hình Thử nghiệm & Thuật toán

### 4.1. Thiết lập Môi trường Kiểm soát

- Tham số ngẫu nhiên: Cố định random seed = 42.
- Phân đồ dữ liệu: Chia tỷ lệ nghiêm ngặt 70% Training (Huấn luyện) và 30% Test (Kiểm tra).
- Kiểm định: Sử dụng k-fold Cross-validation với k = 5 trong quá trình huấn luyện.

### 4.2. Danh sách 10 Thuật toán Bắt buộc

Ngài phải thực hiện thử nghiệm trên toàn bộ 10 thuật toán tương ứng với loại bài toán đã chọn.

#### Đối với Bài toán Phân loại:

- kNN (k-Nearest Neighbors)
- Naive Bayes
- SVM (Support Vector Machine)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- LDA (Linear Discriminant Analysis)
- MLP (Multi-layer Perceptron)
- Logistic Regression

#### Đối với Bài toán Hồi quy:

- Linear Regression
- k-neighbors Regression
- Ridge Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- SGD Regression (Stochastic Gradient Descent)
- Support Vector Regression (SVR)
- Linear SVR
- Multi-layer Perceptron Regression (MLP)

## 5. Chiến lược Tối ưu hóa (Tuning & Optimization)

Việc chỉ chạy thuật toán mặc định là một sự lười biếng không thể chấp nhận. Ngài phải:

- Khảo sát Siêu tham số (Hyper-parameter Tuning): Thực hiện khảo sát cho mọi mô hình (tất cả 10 mô hình), không chỉ riêng những mô hình tốt nhất.
- Phương án cải thiện: Thử nghiệm ít nhất 05 phương án khác nhau để nâng cao hiệu suất (Ví dụ: Tinh chỉnh tham số bằng GridSearch/RandomSearch, áp dụng Ensemble Learning như Stacking, Voting, hoặc thay đổi kỹ thuật Scaling dữ liệu).
- Bằng chứng cải tiến: Phải mô tả rõ ràng sự khác biệt về hiệu suất giữa mô hình cơ sở (Baseline) và mô hình sau khi đã tối ưu.

## 6. Hệ thống Chỉ số Đánh giá (Metrics)

Tất cả số liệu báo cáo phải được làm tròn đến 02 chữ số thập phân.

### 6.1. Bài toán Phân loại

- Các thang đo chính: Accuracy, Precision, Recall, F1-score, và AUC.
- Công cụ phân tích: Bắt buộc sử dụng Ma trận nhầm lẫn (Confusion Matrix) để mổ xẻ các sai sót của mô hình.

### 6.2. Bài toán Hồi quy

- Các thang đo chính: Mean Squared Error (MSE), RMSE, RSE, và Mean Absolute Error (MAE).
- Hiệu suất hệ thống: Bắt buộc báo cáo Thời gian thực thi (Execution Time) cho từng thuật toán để so sánh tính hiệu quả về mặt tài nguyên.

## 7. Cấu trúc Sản phẩm Bàn giao (Deliverables)

Hồ sơ dự án phải được nén thành tệp: SGU-[GroupNo]-[Dataset Name]-Final.zip.

### 7.1. Báo cáo Chi tiết (Report.docx)

Phải tuân thủ cấu trúc 4 phần định sẵn:

- Introduction: Giới thiệu bài toán, ý nghĩa và tập dữ liệu đã chọn.
- Problem Investigation: Trình bày chi tiết EDA. Giải thích rõ lý do thực hiện các bước tiền xử lý (Justification). Nếu không có bằng chứng từ dữ liệu, các bước tiền xử lý được coi là vô nghĩa.
- Experiments and Discussion:
- Bảng so sánh kết quả của 10 thuật toán.
- Xác định 02 thuật toán tốt nhất cho mỗi thang đo (trong 4 thang đo chính) và đưa ra lý giải kỹ thuật tại sao chúng đạt kết quả đó.
- Phân tích sự khác biệt giữa các chỉ số MSE, RMSE, RSE, MAE trong hồi quy.
- Đồ thị minh họa: Vẽ biểu đồ thể hiện sự thay đổi của hàm mất mát (Loss) và độ chính xác (Accuracy) theo thời gian cho cả tập Train và tập Test.
- Conclusions and Future Work: Kết luận kỹ thuật (Technical Conclusion). Tuyệt đối không đưa các cảm nhận cá nhân hoặc trải nghiệm học tập vào phần này.

### 7.2. Slide Thuyết trình (Present.pptx)

Nội dung phải cô đọng và bám sát theo 4 chương của báo cáo (Intro, Investigation, Experiments, Conclusions).

### 7.3. Mã nguồn (Code Folder)

Chứa toàn bộ source code sạch sẽ, tổ chức logic và có chú thích (comments) giải thích các khối lệnh quan trọng.

### 7.4. Thông tin Nhóm (Readme.docx)

Ghi rõ:

- Danh sách thành viên (Họ tên, Mã số sinh viên).
- Bảng phân chia vai trò (Roles): Xác định cụ thể ai làm nhiệm vụ gì. Sự thiếu minh bạch trong phân công sẽ bị đánh giá thấp.
