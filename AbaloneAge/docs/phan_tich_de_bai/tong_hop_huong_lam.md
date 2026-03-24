# Tổng hợp hướng làm cho dự án AbaloneAge

## 1. Kết luận sau khi đọc đề bài và tài liệu tham khảo

Sau khi đối chiếu `final_project.pdf`, các file phân tích đề bài và bộ đồ án mẫu trong `docs/tham_khao`, có thể chốt rằng nhóm cần làm một dự án máy học hoàn chỉnh theo hướng end-to-end, trong đó:

- Có dữ liệu đầu vào rõ ràng.
- Có quá trình phân tích dữ liệu, tiền xử lý, huấn luyện và đánh giá.
- Có báo cáo tiếng Việt.
- Có slide thuyết trình.
- Có phân công công việc từng thành viên.

Điểm quan trọng rút ra từ đồ án mẫu là nhóm không nên chỉ làm một notebook duy nhất, mà cần tách rõ:

- phân tích dữ liệu
- xử lý dữ liệu
- thử nghiệm nhiều mô hình
- tổng hợp kết quả thành báo cáo

## 2. Bài toán của nhóm với bộ dữ liệu Abalone

### 2.1. Mục tiêu bài toán

Bộ dữ liệu `abalone.csv` dùng để dự đoán tuổi của abalone thông qua các đặc trưng đo được từ cơ thể. Trong dữ liệu gốc, biến đích là `Rings`, tức số vòng trên vỏ. Tuổi thực tế thường được suy ra từ công thức:

`Age = Rings + 1.5`

Tuy nhiên trong phần lớn bài toán máy học với bộ dữ liệu này, mô hình sẽ học trực tiếp để dự đoán `Rings`.

### 2.2. Loại bài toán

Đây là **bài toán hồi quy** vì:

- Biến mục tiêu `Rings` là giá trị số nguyên.
- Mục tiêu là dự đoán một đại lượng số chứ không phải nhãn lớp.

### 2.3. Các cột dữ liệu

- `Sex`: giới tính, gồm `M`, `F`, `I`
- `Length`
- `Diameter`
- `Height`
- `WholeWeight`
- `ShuckedWeight`
- `VisceraWeight`
- `ShellWeight`
- `Rings`: biến mục tiêu

## 3. Hướng làm phù hợp nhất cho nhóm

### 3.1. Giai đoạn 1: EDA

Mục tiêu:

- hiểu dữ liệu có gì
- tìm đặc trưng quan trọng
- phát hiện ngoại lệ
- xem phân phối của biến mục tiêu

Cần làm:

- kiểm tra kích thước dữ liệu
- gán tên cột chuẩn
- kiểm tra kiểu dữ liệu
- kiểm tra giá trị thiếu
- thống kê mô tả
- vẽ histogram, boxplot
- phân tích tương quan
- phân tích biến phân loại `Sex`

### 3.2. Giai đoạn 2: Tiền xử lý

Mục tiêu:

- biến dữ liệu thành đầu vào phù hợp cho mô hình

Cần làm:

- encode cột `Sex`
- cân nhắc chuẩn hóa dữ liệu số
- xử lý ngoại lệ nếu cần
- kiểm tra phân phối lệch của các cột trọng lượng
- tạo tập train/test

### 3.3. Giai đoạn 3: Huấn luyện mô hình

Bắt đầu bằng các mô hình cơ bản:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- SVR
- KNN Regressor
- Gradient Boosting Regressor

Nếu còn thời gian có thể mở rộng:

- XGBoost
- AdaBoost Regressor
- MLP Regressor

### 3.4. Giai đoạn 4: Đánh giá

Với bài toán hồi quy, nên dùng:

- MAE
- MSE
- RMSE
- R2

Cần có:

- bảng so sánh metric giữa các mô hình
- biểu đồ so sánh giá trị thực và dự đoán
- nhận xét vì sao chọn mô hình cuối cùng

### 3.5. Giai đoạn 5: Báo cáo và trình bày

Báo cáo cần trả lời rõ:

- bài toán là gì
- dữ liệu gồm những gì
- đã tiền xử lý thế nào
- thử những mô hình nào
- mô hình nào tốt nhất
- hạn chế hiện tại là gì
- hướng phát triển là gì

## 4. Bài học rút ra từ đồ án mẫu trong `docs/tham_khao`

Nhóm mẫu làm tốt ở các điểm sau:

- Có notebook EDA riêng.
- Có nhiều phiên bản thử nghiệm mô hình.
- Có thư mục `src` để chuẩn hóa code.
- Có hình ảnh minh họa cho báo cáo.
- Có paper, slide và poster tách riêng.

Nhóm mình nên học theo:

- không làm dồn hết vào một file
- không chỉ có kết quả cuối, mà phải có quá trình thử nghiệm
- lưu lại biểu đồ và metric để phục vụ viết báo cáo

Nhóm mình không cần sao chép y nguyên:

- cấu trúc pipeline quá phức tạp
- tracking nhiều lớp nếu chưa cần thiết
- quá nhiều mô hình nâng cao khi baseline chưa ổn

## 5. Những việc nên làm ngay

### Việc 1: Hoàn thành notebook EDA đầu tiên

Notebook đầu tiên phải trả lời được:

- dữ liệu có bao nhiêu mẫu
- các cột có ý nghĩa gì
- biến đích phân phối ra sao
- có đặc trưng nào tương quan mạnh với `Rings`
- có ngoại lệ rõ ràng không

### Việc 2: Viết notebook tiền xử lý

Sau EDA, cần chuyển các quyết định xử lý dữ liệu thành notebook riêng:

- encode `Sex`
- chia train/test
- scale dữ liệu nếu cần

### Việc 3: Dựng baseline model

Ít nhất cần có:

- một mô hình tuyến tính
- một mô hình cây
- một mô hình ensemble

### Việc 4: Tổng hợp kết quả để đưa vào báo cáo

Từ đầu nên lưu:

- hình EDA
- bảng metric
- biểu đồ dự đoán

để tránh làm xong mô hình rồi mới quay lại thu thập minh chứng.

## 6. Kết luận ngắn

Hướng làm hợp lý nhất cho dự án này là xem `AbaloneAge` như một bài toán hồi quy chuẩn, làm chắc từ EDA, tiền xử lý, baseline, so sánh mô hình rồi mới chốt mô hình cuối. Đồ án mẫu trong `docs/tham_khao` nên được dùng như tài liệu tham khảo về cách tổ chức và cách trình bày, không nên sao chép nguyên trạng về mặt kỹ thuật.
