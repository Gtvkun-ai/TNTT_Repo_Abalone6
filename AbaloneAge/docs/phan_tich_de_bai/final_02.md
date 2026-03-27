Dưới đây là bản tổng hợp nội dung của tài liệu FINAL PROJECT, đã được bổ sung các ý còn thiếu từ `01_final_project_phan_tich.md` và lược bỏ các phần trùng lặp.

### **1. Mục tiêu của dự án (Objectives)**

Mục tiêu chính của đồ án là sử dụng **scikit-learn** để giải quyết một bài toán học máy thực tế, đồng thời chứng minh năng lực ở hai khía cạnh cốt lõi:

* **Đánh giá hiệu suất (Performance Metrics):** Sử dụng đúng các chỉ số để phản ánh chất lượng mô hình.
* **Tối ưu hóa (Optimization):** Áp dụng các kỹ thuật cải tiến để mô hình sau tối ưu tốt hơn mô hình cơ sở (baseline).

Các mục tiêu cụ thể gồm:

* Thành thạo các thuật toán phân loại và hồi quy trong scikit-learn.
* So sánh hiệu suất của nhiều mô hình trên cùng một bộ dữ liệu.
* Phân tích ưu, nhược điểm của các mô hình dựa trên các thang đo phù hợp.
* Trình bày rõ sự khác biệt giữa mô hình baseline và mô hình sau tối ưu.

### **2. Danh sách bộ dữ liệu (Dataset List)**

Dự án cung cấp 15 bộ dữ liệu, chia thành 3 nhóm bài toán:

| STT | Tên bộ dữ liệu | Loại bài toán |
| :--- | :--- | :--- |
| 1 | Bank Note Authentication | Phân loại nhị phân (Binary Classification) |
| 2 | Horse Colic | Phân loại nhị phân |
| 3 | Sonar | Phân loại nhị phân |
| 4 | German Credit | Phân loại nhị phân |
| 5 | Ionosphere | Phân loại nhị phân |
| 6 | Oil Spill | Phân loại nhị phân |
| 7 | Phoneme | Phân loại nhị phân |
| 8 | Thyroid Gland | Phân loại đa lớp (Multiclass Classification) |
| 9 | Ecoli | Phân loại đa lớp |
| 10 | Glass Identification | Phân loại đa lớp |
| 11 | Wheat Seeds | Phân loại đa lớp |
| 12 | Abalone Age | Hồi quy (Regression) |
| 13 | Wine Quality Red | Hồi quy |
| 14 | Auto Insurance | Hồi quy |
| 15 | Auto Imports Prices | Hồi quy |

### **3. Quy trình thực hiện (Instructions)**

Quy trình triển khai gồm **7 bước bắt buộc**, không được bỏ qua bước nào:

* **Bước 1: Load Data** - Tải và nạp dữ liệu.
* **Bước 2: Initial Data Analysis** - Phân tích sơ bộ về kích thước, kiểu dữ liệu và phân phối cơ bản.
* **Bước 3: Exploratory Data Analysis (EDA)** - Phân tích khám phá dữ liệu để tìm tương quan, outliers, đặc trưng quan trọng và các vấn đề của dữ liệu.
* **Bước 4: Preprocess Data** - Tiền xử lý dữ liệu như xử lý nhiễu, giá trị thiếu, chuẩn hóa, mã hóa, trích xuất hoặc chọn đặc trưng.
* **Bước 5: Build Models** - Xây dựng mô hình trên tập huấn luyện.
* **Bước 6: Evaluate Models** - Đánh giá bằng **k-fold cross-validation**.
* **Bước 7: Assess Model** - Đánh giá cuối cùng trên tập kiểm tra.

Lưu ý quan trọng:

* Mọi bước tiền xử lý hoặc feature engineering phải có **cơ sở từ EDA**, không làm tùy ý.
* Cần giải thích rõ vì sao thực hiện từng bước xử lý dữ liệu.

#### **Thiết lập thực nghiệm**

* Cố định **random seed = 42**.
* Chia dữ liệu theo tỷ lệ **70% train - 30% test**.
* Trong quá trình huấn luyện phải dùng **5-fold cross-validation (k = 5)**.

#### **Danh sách 10 thuật toán bắt buộc**

Với bài toán đã chọn, phải thử nghiệm trên **đủ cả 10 thuật toán tương ứng**.

**Đối với bài toán phân loại:**

* kNN
* Naive Bayes
* SVM
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* LDA
* MLP
* Logistic Regression

**Đối với bài toán hồi quy:**

* Linear Regression
* k-Neighbors Regression
* Ridge Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression
* SGD Regression
* Support Vector Regression (SVR)
* Linear SVR
* MLP Regression

### **4. Đánh giá kết quả và tối ưu hóa**

Tất cả số liệu báo cáo cần được **làm tròn đến 2 chữ số thập phân**.

#### **Đối với bài toán phân loại**

Phải báo cáo kết quả của cả 10 mô hình theo các chỉ số:

* Accuracy
* Precision
* Recall
* F1-score
* AUC

Ngoài ra:

* Bắt buộc sử dụng **Confusion Matrix** để phân tích lỗi của mô hình.
* Cần xác định **2 thuật toán tốt nhất cho từng thang đo chính** và giải thích lý do kỹ thuật.

#### **Đối với bài toán hồi quy**

Phải báo cáo kết quả của cả 10 mô hình theo các chỉ số:

* MSE
* RMSE
* RSE
* MAE
* Execution Time

Ngoài ra:

* Cần phân tích sự khác biệt giữa các chỉ số MSE, RMSE, RSE và MAE.
* Bắt buộc báo cáo **thời gian thực thi** để so sánh hiệu quả tài nguyên giữa các thuật toán.

#### **Yêu cầu tối ưu hóa (Tuning & Optimization)**

Không được chỉ chạy mô hình với tham số mặc định. Cần thực hiện:

* **Khảo sát siêu tham số cho tất cả 10 mô hình**, không chỉ các mô hình tốt nhất.
* Thử nghiệm **ít nhất 5 phương án cải thiện khác nhau**.
* So sánh rõ hiệu suất giữa **baseline** và **mô hình sau tối ưu**.

Ví dụ các hướng cải thiện:

* GridSearchCV
* RandomizedSearchCV
* Thay đổi kỹ thuật scaling
* Feature engineering
* Ensemble learning như Voting hoặc Stacking

### **5. Yêu cầu nộp bài (Submission Requirements)**

Sản phẩm cuối cùng phải được nén thành file:

`SGU-[GroupNo]-[Dataset Name]-Final.zip`

Gói nộp bài cần gồm:

1. **Report.docx**
   
   Báo cáo phải theo đúng 4 phần:
   
   * **Introduction:** Giới thiệu bài toán, ý nghĩa và bộ dữ liệu đã chọn.
   * **Problem Investigation:** Trình bày EDA và giải thích rõ lý do của các bước tiền xử lý. Nếu không có bằng chứng từ dữ liệu, phần tiền xử lý sẽ không có giá trị thuyết phục.
   * **Experiments and Discussion:** So sánh kết quả của 10 thuật toán, mô tả thiết lập thí nghiệm, siêu tham số, baseline và mô hình tối ưu.
   * **Conclusions and Future Work:** Chỉ đưa ra kết luận kỹ thuật và hướng phát triển.

   Bổ sung bắt buộc trong phần **Experiments and Discussion**:
   
   * Bảng so sánh kết quả của 10 thuật toán.
   * Chỉ ra 2 thuật toán tốt nhất cho mỗi thang đo chính.
   * Với bài toán hồi quy, cần phân tích khác biệt giữa MSE, RMSE, RSE, MAE.
   * Có biểu đồ thể hiện sự thay đổi của **loss** và **accuracy** theo thời gian cho cả train và test.

   Lưu ý:
   
   * Không đưa cảm nhận cá nhân hoặc trải nghiệm học tập vào phần kết luận.

2. **Present.pptx**
   
   Slide cần cô đọng và bám theo 4 chương của báo cáo:
   
   * Introduction
   * Investigation
   * Experiments
   * Conclusions

3. **Code Folder**
   
   * Chứa toàn bộ source code sạch, tổ chức logic rõ ràng.
   * Nên có comment cho các khối lệnh quan trọng.

4. **Readme.docx**
   
   Cần ghi rõ:
   
   * Họ tên từng thành viên
   * Mã số sinh viên
   * Vai trò và phần việc của từng người

   Việc phân công không rõ ràng có thể ảnh hưởng đến đánh giá của nhóm.
