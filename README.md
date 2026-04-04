# AbaloneAge


Dự án môn Machine Learning cho bài toán dự đoán số vòng `Rings` của abalone nhằm ước lượng tuổi từ các đặc trưng hình thái và khối lượng cơ thể. Đây là dự án chính trong repository `TNTT_Repo_Abalone6`.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Mục tiêu](#mục-tiêu)
- [Dữ liệu](#dữ-liệu)
- [Quy trình thực hiện](#quy-trình-thực-hiện)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt và chạy dự án](#cài-đặt-và-chạy-dự-án)
- [Các thực nghiệm hiện có](#các-thực-nghiệm-hiện-có)
- [Kết quả và đầu ra](#kết-quả-và-đầu-ra)
- [Tài liệu báo cáo](#tài-liệu-báo-cáo)
- [Phân công công việc](#phân-công-công-việc)

## Tổng quan

Trong thực tế, tuổi của abalone thường được xác định bằng cách cắt vỏ, nhuộm và đếm số vòng dưới kính hiển vi. Cách tiếp cận này tốn thời gian và khó triển khai trên quy mô lớn. Dự án `AbaloneAge` xây dựng một pipeline học máy để dự đoán `Rings` từ các đặc trưng dễ thu thập hơn như kích thước và khối lượng.

Theo tài liệu của bộ dữ liệu:

```text
Age ≈ Rings + 1.5
```

Trong phạm vi repo này, nhóm dự đoán trực tiếp biến mục tiêu `Rings` dưới dạng bài toán hồi quy.

## Mục tiêu

- Hiểu cấu trúc và đặc điểm của bộ dữ liệu Abalone.
- Xây dựng pipeline tiền xử lý và huấn luyện mô hình rõ ràng, dễ tái sử dụng.
- So sánh nhiều chiến lược preprocessing và feature engineering.
- Đánh giá mô hình bằng các metric phù hợp cho bài toán hồi quy.
- Chuẩn bị kết quả phục vụ báo cáo, poster và slide thuyết trình.

## Dữ liệu

Nguồn dữ liệu chính nằm tại `AbaloneAge/data/raw/abalone.csv`.

Thông tin cơ bản:

- `4177` mẫu dữ liệu.
- `8` thuộc tính đầu vào.
- `1` biến mục tiêu là `Rings`.

Các cột chính:

- `Sex`
- `Length`
- `Diameter`
- `Height`
- `WholeWeight`
- `ShuckedWeight`
- `VisceraWeight`
- `ShellWeight`
- `Rings`

Một số quan sát quan trọng từ phần EDA và báo cáo:

- dữ liệu không có missing value đáng kể,
- có một biến phân loại là `Sex`,
- các biến khối lượng và mục tiêu có độ lệch phân phối,
- tồn tại outlier ở một số thuộc tính,
- quan hệ giữa đầu vào và `Rings` không hoàn toàn tuyến tính.

## Quy trình thực hiện

Luồng làm việc trong project được triển khai theo các notebook và module trong `src/`:

1. `Load Data`
   Đọc dữ liệu từ `data/raw/abalone.csv`.
2. `Initial Analysis`
   Kiểm tra kích thước dữ liệu, kiểu dữ liệu, missing values và duplicate records.
3. `EDA`
   Phân tích phân phối biến mục tiêu, outlier, tương quan và ảnh hưởng của `Sex`.
4. `Preprocessing`
   Chuẩn hóa tên cột, mã hóa biến phân loại, scaling cho biến số và chia train/test.
5. `Modeling`
   Huấn luyện baseline và thực nghiệm trên nhiều cấu hình dữ liệu.
6. `Evaluation`
   Đánh giá mô hình bằng `MAE`, `MSE`, `RMSE`, `RSE`, `R2`.
7. `Experimentation`
   So sánh giữa baseline, preprocessing khác nhau và feature engineering thủ công.

## Cấu trúc thư mục

```text
TNTT_Repo_Abalone6/
|-- README.md
|-- AbaloneAge/
|   |-- data/
|   |   |-- raw/
|   |   |-- interim/
|   |   `-- processed/
|   |-- docs/
|   |-- experiments/
|   |-- notebooks/
|   |-- outputs/
|   |-- reports/
|   |-- src/
|   |   |-- config/
|   |   |-- data/
|   |   |-- features/
|   |   |-- models/
|   |   |-- utils/
|   |   `-- visualization/
|   |-- tests/
|   `-- requirements.txt
`-- pima-indian-diabetes/
```

Thư mục quan trọng:

- `AbaloneAge/src/data/`: load, clean và chia dữ liệu.
- `AbaloneAge/src/features/`: encoding, scaling và feature engineering.
- `AbaloneAge/src/models/`: train, predict, evaluate.
- `AbaloneAge/notebooks/`: notebook phân tích và thử nghiệm theo từng giai đoạn.
- `AbaloneAge/experiments/`: script thực nghiệm độc lập để xuất kết quả JSON.
- `AbaloneAge/reports/`: báo cáo tiếng Việt, poster, slide và phân công.

## Công nghệ sử dụng

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- pytest

## Cài đặt và chạy dự án

Di chuyển vào thư mục dự án:

```powershell
cd AbaloneAge
```

Tạo và kích hoạt môi trường ảo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Cài đặt thư viện:

```powershell
pip install -r requirements.txt
```

Mở notebook để làm việc theo từng bước:

```powershell
jupyter notebook
```

Thứ tự notebook nên xem:

1. `notebooks/01_kham_pha_du_lieu.ipynb`
2. `notebooks/02_tien_xu_ly_du_lieu.ipynb`
3. `notebooks/03_huan_luyen_mo_hinh.ipynb`
4. `notebooks/04_danh_gia_mo_hinh.ipynb`
5. `notebooks/05_thu_nghiem_bo_sung.ipynb`

Chạy các thực nghiệm bằng script:

```powershell
python experiments/01_baseline_random_forest.py
python experiments/02_preprocessing_comparison.py
python experiments/03_feature_engineering_ablation.py
```

Chạy test:

```powershell
pytest
```

## Các thực nghiệm hiện có

### 1. Baseline Random Forest

File: `experiments/01_baseline_random_forest.py`

- huấn luyện `RandomForestRegressor` baseline,
- so sánh với một cấu hình tuned,
- lưu kết quả để đưa vào phần so sánh mô hình.

### 2. So sánh preprocessing

File: `experiments/02_preprocessing_comparison.py`

- `linear_encoded_only`
- `linear_standard_scaled`
- `rf_robust_scaled`

Mục tiêu là kiểm tra mức độ phù hợp giữa kiểu tiền xử lý và loại mô hình.

### 3. Feature engineering ablation

File: `experiments/03_feature_engineering_ablation.py`

Các đặc trưng thủ công đã thử:

- `VolumeApprox`
- `TotalWeightToShell`
- `EdibleWeightRatio`

Mục tiêu là so sánh dữ liệu gốc với dữ liệu đã được enrich feature.

## Kết quả và đầu ra

Kết quả hiện tại của dự án được lưu tại:

- `AbaloneAge/experiments/results/`
- `AbaloneAge/outputs/metrics/`
- `AbaloneAge/outputs/models/`
- `AbaloneAge/outputs/figures/`
- `AbaloneAge/outputs/logs/`

Các metric được sử dụng trong repo:

- `MAE`
- `MSE`
- `RMSE`
- `RSE`
- `R2`

Theo nội dung báo cáo hiện có trong repo, dự án đã ghi nhận các hướng so sánh giữa:

- baseline model,
- optimized model,
- feature engineering,
- preprocessing variants,
- ensemble approaches.

## Tài liệu báo cáo

Các tài nguyên phục vụ nộp bài nằm trong:

- `AbaloneAge/reports/bao_cao_tieng_viet/report.md`
- `AbaloneAge/reports/poster/`
- `AbaloneAge/reports/slides/presentation.md`
- `AbaloneAge/docs/`

Nếu cần chuẩn bị bản nộp cuối, nên đọc theo thứ tự:

1. `docs/` để nắm đề bài và định hướng.
2. `notebooks/` để theo dõi quy trình thực hiện.
3. `experiments/results/` để lấy số liệu.
4. `reports/` để tổng hợp báo cáo, poster và slide.

## Phân công công việc

Theo file `AbaloneAge/reports/phan_cong_cong_viec/team_assignment.md`, hiện repo đang để sẵn khung phân công:

| STT | Thành viên            | Nhiệm vụ chính |
|-----|-----------------------|----------------|
| 1   | Cao Nguyễn Yên Hòa    | Lập kế hoạch; nghiên cứu tài liệu (Paper); xây dựng chuỗi 5 notebook nền tảng (01–05: khám phá dữ liệu, tiền xử lý, huấn luyện, đánh giá, thử nghiệm bổ sung). |
| 2   | Trần Nguyễn Đăng Khoa | Nội dung Report; phân tích tương quan (EDA correlation); tiền xử lý ngoại lệ nâng cao (Preprocessing outlier v2); chọn lọc đặc trưng dựa trên mô hình (Feature selection model-based); module so sánh toàn diện (Model compare v2). |
| 3   | Lưu Gia Bảo           | Thiết kế và chỉnh sửa nội dung Report; EDA tổng quan nâng cao (data overview v2); mã hóa đặc trưng (Preprocessing encoding); chọn lọc đặc trưng thống kê (Feature selection KBest); triển khai mô hình Cây quyết định (Tree models); đánh giá chuyên sâu 2 mô hình xuất sắc nhất (Final best 2 models). |
| 4   | Lâm Hán Đạt           | Nội dung Slide; EDA tổng quan dữ liệu (data overview); module tiền xử lý cơ sở (Preprocessing baseline); chọn lọc đặc trưng cơ sở (Feature selection baseline); triển khai nhóm mô hình Tuyến tính (Linear models); xác lập mốc đánh giá cuối cùng (Final baseline). |
| 5   | Hồ Trung Tín          | Thiết kế Slide; rà soát ngoại lệ (EDA outlier check) và chuẩn hóa dữ liệu (Scaling); chọn lọc đặc trưng bằng RFE (baseline, compare v2); thử nghiệm chiến lược học tích hợp (Ensemble try 1); đóng gói kết quả toàn dự án (Report summary). |


## Ghi chú

- Dự án chính trong repo là `AbaloneAge/`, còn `pima-indian-diabetes/` là thư mục bài tập riêng.
- Bài toán hiện tại là hồi quy trên biến mục tiêu `Rings`.
- README này được viết để giúp đọc nhanh repo, onboard thành viên mới và hỗ trợ chuẩn bị báo cáo đồ án.
