# Phân tích tài liệu `hand_gesture_recognition_slides.pdf`

## 1. Vai trò của bộ slide

Bộ slide này không chỉ tóm tắt bài báo mà còn cho thấy cách trình bày đề tài theo hướng học thuật: đặt vấn đề, mô tả dữ liệu, trình bày quy trình, đưa ra kết quả và kết luận. Đây là tài liệu rất hữu ích cho phần bảo vệ của nhóm.

## 2. Nội dung chính được trình bày trong slide

### 2.1. Bài toán và động cơ nghiên cứu

- Bài toán là nhận dạng cử chỉ tay động.
- Mục tiêu ứng dụng trong tương tác người máy.
- Khó khăn đến từ kích thước bàn tay nhỏ, cử chỉ phức tạp và self-occlusion.

### 2.2. Bộ dữ liệu

- Sử dụng `DHG Dataset`.
- Có 14 loại cử chỉ.
- Mỗi cử chỉ được thu từ nhiều người dùng khác nhau.

### 2.3. Quy trình xử lý

Slide thể hiện quy trình thành các bước rõ ràng:

1. Trích xuất đặc trưng từ depth và skeleton.
2. Tạo visual words bằng `k-means` hoặc `GMM`.
3. Xây dựng vector đặc trưng theo `Temporal Pyramid`.
4. Đưa vào `SVM` để phân loại.
5. Thử nghiệm kết hợp nhiều mô hình bằng fusion.

## 3. Giá trị của slide đối với nhóm

### 3.1. Giá trị về nội dung

- Giúp nhóm nhìn nhanh toàn bộ pipeline mà không cần đọc lại toàn bộ bài báo.
- Dễ nhận ra thành phần nào là cốt lõi của phương pháp.
- Hữu ích khi tổng hợp ý để viết phần mô tả phương pháp trong báo cáo.

### 3.2. Giá trị về cách trình bày

Slide cho thấy một cấu trúc thuyết trình hợp lý:

1. Giới thiệu bài toán
2. Khó khăn và động lực nghiên cứu
3. Dữ liệu
4. Phương pháp đề xuất
5. Kết quả thực nghiệm
6. Kết luận

Đây cũng chính là khung mà nhóm có thể tái sử dụng cho file slide bảo vệ của mình.

## 4. Kết quả thực nghiệm được nhấn mạnh trong slide

| Phương pháp | Đặc trưng | Clustering | Accuracy |
| --- | --- | --- | --- |
| 1 | Skeleton | K-means | 53.39% |
| 2 | Skeleton | GMM | 55.18% |
| 3 | Depth | K-means | 76.25% |
| 4 | Depth | GMM | 71.96% |
| 5 | Fusion (Max) | Kết hợp | 80.89% |
| 6 | Fusion (Average) | Kết hợp | 82.50% |
| 7 | Fusion (Weighted) | Kết hợp có trọng số | 83.75% |

Nhận xét:
- Slide nhấn mạnh rằng fusion cho kết quả tốt hơn từng kênh riêng lẻ.
- Weighted fusion là phương án tốt nhất trong các cách kết hợp được thử nghiệm.

## 5. Điều nhóm nên học theo từ slide

- Khi bảo vệ, cần có sơ đồ pipeline rõ ràng.
- Nên đưa bảng kết quả để so sánh thay vì nói bằng lời.
- Nên chốt một thông điệp chính ở phần kết luận, ví dụ: "kết hợp đặc trưng giúp cải thiện kết quả".
- Mỗi slide chỉ nên chứa một ý chính, tránh nhiều chữ và mạnh ai nấy nói.

## 6. Đề xuất áp dụng vào file `reports/slides`

Nhóm có thể xây dựng slide theo thứ tự sau:

1. Tên đề tài và thành viên
2. Bài toán và mục tiêu
3. Mô tả dữ liệu
4. Tiền xử lý dữ liệu
5. Mô hình và quy trình huấn luyện
6. Kết quả và so sánh
7. Kết luận và hướng phát triển
8. Phần hỏi đáp

## 7. Kết luận ngắn

Nếu bài báo là tài liệu học thuật gốc, thì bộ slide là tài liệu cho thấy cách đóng gói nội dung học thuật thành một bài thuyết trình mạch lạc. Đây là nguồn tham khảo tốt để nhóm xây dựng slide bảo vệ vừa ngắn gọn vừa có đủ cơ sở học thuật.
