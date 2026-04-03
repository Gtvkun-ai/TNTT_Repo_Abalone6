# TÓM TẮT

Trong báo cáo này, chúng tôi nghiên cứu bài toán dự đoán số vòng tuổi `Rings` của bào ngư từ các đặc trưng hình thái và khối lượng trong bộ dữ liệu Abalone thuộc UCI Machine Learning Repository. Bộ dữ liệu gồm `4177` mẫu, `8` đặc trưng đầu vào và một biến mục tiêu, trong đó tuổi thực tế có thể được suy ra gần đúng theo công thức `Age = Rings + 1.5`. Mục tiêu của đồ án là xây dựng một quy trình thực nghiệm hoàn chỉnh cho bài toán hồi quy, bao gồm phân tích dữ liệu, tiền xử lý, so sánh mô hình và đánh giá tác động của các bước cải tiến.

Trên cơ sở tham khảo cách tổ chức thực nghiệm của paper Music Genre Classification, báo cáo được triển khai theo hướng tăng dần độ phức tạp: xây dựng baseline, so sánh các chiến lược tiền xử lý, đánh giá feature engineering và chuẩn bị nền tảng cho các bước tuning, benchmark nhiều mô hình và ensemble. Kết quả thực nghiệm ban đầu cho thấy tuning Random Forest giúp giảm `RMSE` từ `2.1946` xuống `2.1717`, trong khi thay đổi preprocessing chỉ tạo ra khác biệt nhỏ. Đáng chú ý hơn, việc bổ sung các đặc trưng thủ công như `VolumeApprox`, `TotalWeightToShell` và `EdibleWeightRatio` giúp giảm `RMSE` từ `2.1844` xuống `2.1542`, cho thấy chất lượng biểu diễn đặc trưng có vai trò quan trọng hơn chỉ thay đổi phép chuẩn hóa.

Những kết quả này gợi ý rằng với dữ liệu Abalone, hướng cải thiện hiệu năng hiệu quả không nằm ở việc xử lý dữ liệu theo cách quá phức tạp, mà nằm ở việc khai thác tốt hơn ý nghĩa sinh học và hình học của các biến đầu vào. Từ đó, báo cáo đề xuất một khung thực nghiệm có thể mở rộng cho các bước benchmark, tối ưu hóa và kết hợp mô hình ở các giai đoạn tiếp theo.

# LỜI MỞ ĐẦU

Trong sinh học biển và quản lý nguồn lợi thủy sản, việc xác định độ tuổi của bào ngư có ý nghĩa quan trọng đối với nghiên cứu tăng trưởng, cấu trúc quần thể, khả năng tái tạo và khai thác bền vững. Tuy nhiên, tuổi của bào ngư không thể xác định chính xác chỉ bằng quan sát kích thước bên ngoài, vì tốc độ sinh trưởng của chúng còn chịu ảnh hưởng bởi môi trường sống, nguồn thức ăn và điều kiện sinh thái.

Theo mô tả của bộ dữ liệu Abalone trên UCI Machine Learning Repository, tuổi của bào ngư thường được xác định bằng cách cắt vỏ qua phần chóp, nhuộm mẫu và đếm số vòng sinh trưởng dưới kính hiển vi. Đây là một quy trình tốn thời gian, mang tính phá hủy mẫu và khó mở rộng cho số lượng lớn. Vì vậy, việc dự đoán tuổi từ các phép đo vật lý như chiều dài, đường kính, chiều cao và các loại khối lượng trở thành một bài toán học máy có giá trị thực tiễn rõ rệt.

Báo cáo này tập trung vào bài toán dự đoán số vòng tuổi `Rings` của bào ngư từ các đặc trưng hình thái và khối lượng trong bộ dữ liệu Abalone. Trên cơ sở đó, nhóm hướng tới ba mục tiêu chính: phân tích đặc điểm dữ liệu, xây dựng các mô hình hồi quy phù hợp và đánh giá mức độ hiệu quả của hướng tiếp cận mô hình hóa dữ liệu so với cách xác định tuổi truyền thống. Đồng thời, báo cáo cũng tổ chức phần thực nghiệm theo từng giả thuyết cải tiến, thay vì chỉ liệt kê rời rạc các thuật toán.

Điểm quan trọng cần làm rõ ngay từ đầu là bộ dữ liệu Abalone trên UCI không phải một "paper ML chính thức", mà là dữ liệu được lưu trữ lại từ một nghiên cứu sinh học thực tế. Theo thông tin trích dẫn của UCI, dữ liệu gắn với công trình của Warwick J. Nash và cộng sự; còn nghiên cứu gốc về mặt sinh học là báo cáo *The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone (H. rubra) from the North Coast and Islands of Bass Strait* xuất bản năm 1994. Việc nêu đúng nguồn gốc này giúp báo cáo có cơ sở học thuật rõ ràng hơn khi trình bày bài toán.

# CHƯƠNG 1: GIỚI THIỆU ĐỒ ÁN

## 1.1. Bối cảnh và ý nghĩa bài toán

Bào ngư là loài thân mềm chân bụng sống ở biển, có giá trị kinh tế cao trong lĩnh vực thủy sản và chế biến thực phẩm. Ngoài giá trị thương mại, bào ngư còn là đối tượng nghiên cứu quan trọng trong sinh học biển do đặc điểm tăng trưởng phụ thuộc mạnh vào điều kiện môi trường. Vì vậy, việc ước lượng tuổi của bào ngư không chỉ phục vụ mục tiêu khai thác mà còn hỗ trợ theo dõi quần thể và quản lý nguồn lợi một cách bền vững.

Trong thực tế, phương pháp xác định tuổi đáng tin cậy nhất là đếm số vòng sinh trưởng trên vỏ. Tuy nhiên, đây là quy trình thủ công, tốn công, đòi hỏi thao tác cắt, mài, nhuộm và quan sát dưới kính hiển vi. Khi số lượng mẫu lớn, phương pháp này trở nên kém hiệu quả về mặt chi phí và thời gian. Từ đó, bài toán dự đoán tuổi bào ngư từ các đặc trưng vật lý trở thành một hướng tiếp cận phù hợp để ứng dụng học máy vào thực tế.

## 1.2. Nguồn gốc bộ dữ liệu Abalone

Bộ dữ liệu được sử dụng trong đồ án là **Abalone** từ **UCI Machine Learning Repository**. UCI mô tả đây là bộ dữ liệu dùng để dự đoán tuổi của bào ngư từ các phép đo vật lý, với:

- `4177` mẫu quan sát
- `8` đặc trưng đầu vào
- `1` biến mục tiêu là `Rings`
- không có giá trị thiếu sau khi làm sạch dữ liệu gốc

Theo mô tả chính thức của UCI, tuổi bào ngư được suy ra theo công thức:

`Age = Rings + 1.5`

Điều này có nghĩa là `Rings` là biến đích trực tiếp trong bài toán, còn tuổi thực tế theo năm có thể được ước lượng bằng cách cộng thêm `1.5`.

Về mặt học thuật, cần phân biệt hai lớp nguồn:

- **Nguồn lưu trữ dữ liệu cho cộng đồng học máy**: UCI Machine Learning Repository, với trích dẫn chuẩn là `Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55C7W`.
- **Nguồn gốc sinh học của dữ liệu**: nghiên cứu của Warwick J. Nash và cộng sự về quần thể bào ngư ở Tasmania, công bố năm 1994.

Việc xác định đúng nguồn gốc dataset là cần thiết vì nhiều người thường nhầm rằng Abalone có một "paper ML gốc", trong khi thực tế đây là bộ dữ liệu sinh học được cộng đồng học máy sử dụng như một benchmark kinh điển cho bài toán hồi quy và phân loại.

## 1.3. Phát biểu bài toán

Trong đồ án này, nhóm xem bài toán AbaloneAge là một **bài toán hồi quy**, trong đó:

- đầu vào là các phép đo vật lý của bào ngư
- đầu ra là số vòng tuổi `Rings`

So với cách tiếp cận phân loại theo nhóm tuổi, bài toán hồi quy phù hợp hơn với bản chất liên tục của biến mục tiêu và giúp giữ lại nhiều thông tin hơn trong quá trình học. Sau khi dự đoán `Rings`, có thể quy đổi xấp xỉ sang tuổi thật bằng công thức `Age = Rings + 1.5`.

## 1.4. Hai hướng tiếp cận xác định tuổi bào ngư

### 1.4.1. Hướng thực nghiệm sinh học

Đây là phương pháp truyền thống, đồng thời là cơ sở để tạo ra giá trị mục tiêu cho bộ dữ liệu. Quy trình tổng quát gồm:

1. Thu thập mẫu bào ngư.
2. Cắt vỏ qua phần chóp.
3. Mài và nhuộm mẫu để làm rõ các vòng sinh trưởng.
4. Quan sát dưới kính hiển vi.
5. Đếm số vòng và suy ra tuổi.

Ưu điểm của hướng này là có độ tin cậy sinh học cao. Tuy nhiên, nhược điểm là tốn nhiều thời gian, chi phí lớn, mang tính phá hủy mẫu và khó áp dụng trên quy mô rộng.

### 1.4.2. Hướng mô hình hóa dữ liệu và học máy

Đây là hướng được lựa chọn trong báo cáo. Thay vì can thiệp vật lý vào mẫu vật, mô hình học máy sử dụng các biến đo đạc sẵn có như kích thước và khối lượng để học mối liên hệ với `Rings`. Hướng tiếp cận này có các ưu điểm:

- không phá hủy mẫu vật
- dự đoán nhanh sau khi mô hình được huấn luyện
- chi phí triển khai thấp hơn
- dễ mở rộng cho số lượng lớn mẫu dữ liệu

Dĩ nhiên, chất lượng dự đoán phụ thuộc mạnh vào dữ liệu đầu vào, quy trình tiền xử lý và mức độ phù hợp của mô hình. Vì vậy, báo cáo không chỉ xây dựng mô hình mà còn phải phân tích dữ liệu và thử nghiệm nhiều chiến lược cải thiện hiệu năng.

### 1.4.3. So sánh ngắn gọn hai hướng tiếp cận

| Tiêu chí | Thực nghiệm sinh học | Mô hình hóa dữ liệu |
|---|---|---|
| Độ chính xác sinh học | Cao | Phụ thuộc vào mô hình |
| Tốc độ xử lý | Chậm | Nhanh |
| Chi phí triển khai | Cao | Thấp hơn |
| Tính bảo toàn mẫu | Phá hủy mẫu | Không phá hủy mẫu |
| Khả năng mở rộng | Hạn chế | Tốt |

Từ so sánh trên, nhóm lựa chọn hướng mô hình hóa dữ liệu làm trọng tâm triển khai của đồ án.

## 1.5. Mục tiêu và đóng góp chính của đồ án

Bám theo yêu cầu tổng quát của báo cáo, đồ án hướng tới các mục tiêu sau:

- phân tích đặc điểm của bộ dữ liệu Abalone và làm rõ bản chất bài toán
- xây dựng pipeline tiền xử lý phù hợp cho dữ liệu bảng có cả biến số và biến phân loại
- thử nghiệm các mô hình hồi quy và so sánh hiệu năng
- đánh giá tác động của các bước cải tiến như preprocessing, feature engineering, tuning và ensemble
- rút ra nhận xét về yếu tố nào thực sự giúp cải thiện kết quả dự đoán

Theo định hướng này, phần thực nghiệm của báo cáo sẽ được tổ chức như một chuỗi giả thuyết kiểm chứng dần, thay vì chỉ liệt kê mô hình. Cách trình bày này cũng phù hợp với tinh thần của paper tham khảo mà nhóm đang học theo.

Ở giai đoạn hiện tại, báo cáo đã ghi nhận một số đóng góp ban đầu:

- xây dựng được mốc baseline bằng Random Forest để làm cơ sở so sánh
- kiểm tra ảnh hưởng của các chiến lược preprocessing và nhận thấy mức cải thiện còn hạn chế
- chứng minh được rằng feature engineering thủ công mang lại cải thiện rõ hơn preprocessing thuần túy
- tạo nền kết quả định lượng để phát triển tiếp các phần benchmark nhiều mô hình, tuning sâu hơn và ensemble

## 1.6. Cấu trúc của báo cáo

Phần còn lại của báo cáo được tổ chức như sau:

- Chương 2 trình bày phân tích bài toán và mô tả bộ dữ liệu Abalone.
- Chương 3 trình bày phương pháp và quá trình thực nghiệm theo từng phiên bản cải tiến.
- Chương 4 tổng hợp kết quả, so sánh và thảo luận các phát hiện chính.
- Chương 5 nêu kết luận và các hướng cải tiến tiếp theo.

# CHƯƠNG 2: PHÂN TÍCH BÀI TOÁN VÀ DỮ LIỆU

## 2.1. Mô tả bộ dữ liệu

Bộ dữ liệu Abalone gồm 8 thuộc tính đầu vào và 1 biến mục tiêu. Các biến được sử dụng trong đồ án bao gồm:

| Biến | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `Sex` | Phân loại | Giới tính với ba giá trị `M`, `F`, `I` |
| `Length` | Liên tục | Chiều dài lớn nhất của vỏ (mm) |
| `Diameter` | Liên tục | Đường kính vuông góc với chiều dài (mm) |
| `Height` | Liên tục | Chiều cao của bào ngư khi còn trong vỏ (mm) |
| `WholeWeight` | Liên tục | Khối lượng toàn phần (gram) |
| `ShuckedWeight` | Liên tục | Khối lượng phần thịt (gram) |
| `VisceraWeight` | Liên tục | Khối lượng nội tạng sau khi làm sạch (gram) |
| `ShellWeight` | Liên tục | Khối lượng vỏ khô sau khi sấy (gram) |
| `Rings` | Số nguyên | Biến mục tiêu, dùng để suy ra tuổi |

Từ mô tả chính thức của UCI và hướng phân tích của đồ án, có thể rút ra một số nhận định ban đầu:

- dữ liệu không có giá trị thiếu
- chỉ có một biến phân loại là `Sex`, còn lại chủ yếu là biến số liên tục
- các biến về khối lượng và kích thước có khả năng tương quan với `Rings`
- mối quan hệ giữa đầu vào và biến mục tiêu có thể không hoàn toàn tuyến tính
- một số biến có thể xuất hiện độ lệch phân phối hoặc ngoại lệ, nên cần được kiểm tra ở bước EDA

Những nhận định này là cơ sở để nhóm triển khai các bước tiếp theo như trực quan hóa dữ liệu, so sánh chiến lược tiền xử lý và xây dựng đặc trưng mới.

## 2.2. Những thách thức chính của bài toán

So với nhiều bài toán hồi quy chuẩn trên dữ liệu bảng, bài toán AbaloneAge có một số khó khăn đáng chú ý:

- số lượng đặc trưng không lớn, nên hiệu quả mô hình phụ thuộc mạnh vào chất lượng biểu diễn của từng biến
- dữ liệu có sự kết hợp giữa biến phân loại và biến số liên tục, nên cần lựa chọn pipeline tiền xử lý phù hợp
- các biến khối lượng và kích thước có thể tương quan mạnh với nhau, làm xuất hiện hiện tượng dư thừa thông tin
- biến mục tiêu `Rings` là số nguyên nhưng được dự đoán dưới góc nhìn hồi quy, nên mô hình cần vừa học tốt xu hướng chung vừa giữ sai số ở mức thấp
- bài toán có khả năng chứa quan hệ phi tuyến giữa kích thước, khối lượng và tuổi, khiến các mô hình tuyến tính đơn giản có thể bị giới hạn

Từ các đặc điểm trên, báo cáo ưu tiên đánh giá mô hình bằng các chỉ số sai số như `RMSE`, `MAE`, `MSE` và `R2`, đồng thời phân tích riêng tác động của preprocessing và feature engineering lên chất lượng dự đoán.

## 2.3. Định hướng phân tích dữ liệu khám phá

Ở giai đoạn EDA, báo cáo sẽ tập trung trả lời các câu hỏi sau:

1. Phân phối của biến mục tiêu `Rings` có trải đều hay tập trung ở một vài mức tuổi?
2. Biến `Sex` có ảnh hưởng đáng kể đến `Rings` hay không?
3. Các biến kích thước và khối lượng có mối tương quan tuyến tính mạnh đến đâu với biến mục tiêu?
4. Có tồn tại ngoại lệ hoặc phân phối lệch ở các biến đầu vào không?
5. Feature engineering có thể khai thác thêm thông tin gì từ các biến vật lý gốc?

Từ đây, phần thực nghiệm ở các chương sau sẽ được tổ chức theo hướng:

- xây dựng mốc baseline
- so sánh preprocessing
- kiểm tra hiệu quả của feature engineering
- mở rộng sang nhiều mô hình hồi quy
- tối ưu và kết hợp mô hình khi cần thiết

Theo kết quả hiện có trong repo, ba phát hiện định lượng đầu tiên đã định hình hướng triển khai tiếp theo:

- tuning Random Forest giúp giảm `RMSE` từ `2.1946` xuống `2.1717`
- thay đổi preprocessing chỉ tạo ra khác biệt nhỏ, với `RMSE` dao động quanh `2.185 - 2.187`
- feature engineering giúp giảm `RMSE` từ `2.1844` xuống `2.1542`, tốt hơn mức cải thiện thu được từ preprocessing

Điều này cho thấy phần tiếp theo của báo cáo nên ưu tiên khai thác đặc trưng và so sánh mô hình trên cùng một khung đánh giá nhất quán, thay vì kỳ vọng quá nhiều vào riêng bước chuẩn hóa dữ liệu.

## 2.4. Kết quả khám phá dữ liệu từ notebook 01

Dựa trên notebook `01_kham_pha_du_lieu.ipynb`, nhóm thu được các kết quả EDA quan trọng như sau.

### 2.4.1. Kích thước và cấu trúc dữ liệu

Tập dữ liệu sau khi nạp có kích thước `4177` dòng và `9` cột. Trong đó:

- `1` biến phân loại: `Sex`
- `7` biến số thực: `Length`, `Diameter`, `Height`, `WholeWeight`, `ShuckedWeight`, `VisceraWeight`, `ShellWeight`
- `1` biến mục tiêu nguyên: `Rings`

Toàn bộ các cột đều có đủ `4177` giá trị hợp lệ. Notebook cũng kiểm tra lại chất lượng dữ liệu và cho thấy:

- tổng số giá trị thiếu: `0`
- số dòng trùng lặp: `0`

Kết quả này rất thuận lợi cho bài toán hồi quy, vì nhóm không cần xử lý missing value hay loại bỏ nhiều điểm dữ liệu trước khi huấn luyện.

### 2.4.2. Phân phối biến mục tiêu `Rings`

Thống kê mô tả của biến mục tiêu trong notebook cho thấy:

- giá trị trung bình: `9.9337`
- trung vị: `9`
- độ lệch chuẩn: `3.2242`
- giá trị nhỏ nhất: `1`
- giá trị lớn nhất: `29`
- độ lệch phân phối `skewness`: `1.1141`

Như vậy, `Rings` có xu hướng lệch phải nhẹ, nghĩa là phần lớn mẫu tập trung ở vùng tuổi thấp đến trung bình, trong khi vẫn tồn tại một nhóm cá thể có số vòng khá cao. Đây là một dấu hiệu quan trọng, vì nó cho thấy mô hình cần dự đoán tốt ở vùng trung tâm nhưng vẫn không được bỏ qua các cá thể già hơn nằm ở phần đuôi phân phối.

### 2.4.3. Vai trò của biến `Sex`

Notebook EDA đã phân tích tần suất và phân phối `Rings` theo từng nhóm `Sex`. Kết quả tổng hợp như sau:

| Nhóm `Sex` | Số mẫu | Mean `Rings` | Median `Rings` | Std |
|---|---:|---:|---:|---:|
| `F` | 1307 | 11.13 | 10 | 3.10 |
| `I` | 1342 | 7.89 | 8 | 2.51 |
| `M` | 1528 | 10.71 | 10 | 3.03 |

Từ bảng trên có thể thấy:

- nhóm `I` có số vòng trung bình thấp nhất, phù hợp với ý nghĩa "infant"
- nhóm `F` có số vòng trung bình cao nhất trong ba nhóm
- sự khác biệt này cho thấy `Sex` không phải một biến dư thừa, mà có khả năng mang thông tin dự báo hữu ích

Vì vậy, quyết định giữ lại và mã hóa `Sex` trong pipeline là hoàn toàn hợp lý.

### 2.4.4. Ngoại lệ và độ lệch phân phối

Nhóm sử dụng boxplot và quy tắc IQR để nhận diện ngoại lệ. Kết quả cho thấy các biến đều có một tỷ lệ outlier nhất định, trong đó nổi bật là:

| Biến | Tỷ lệ outlier |
|---|---:|
| `Rings` | 6.66% |
| `Diameter` | 1.41% |
| `Length` | 1.17% |
| `ShuckedWeight` | 1.15% |
| `ShellWeight` | 0.84% |
| `WholeWeight` | 0.72% |
| `Height` | 0.69% |
| `VisceraWeight` | 0.62% |

Nhận xét từ EDA cho thấy:

- outlier không quá cực đoan ở từng biến đầu vào, nhưng xuất hiện ở hầu hết các biến quan trọng
- các biến khối lượng có độ phân tán tương đối lớn
- biến `Height` có giá trị nhỏ nhất bằng `0.0`, cần được quan sát cẩn thận vì về trực giác đây là giá trị bất thường

Thay vì loại bỏ outlier ngay từ đầu, nhóm chọn chiến lược thận trọng hơn: giữ nguyên dữ liệu gốc cho baseline, đồng thời tạo thêm một phiên bản dữ liệu `robust_log_scaled` để kiểm tra xem log-transform và `RobustScaler` có cải thiện chất lượng dự đoán hay không.

### 2.4.5. Tương quan giữa các biến với `Rings`

Ma trận tương quan trong notebook chỉ ra các biến có quan hệ mạnh nhất với `Rings` như sau:

| Biến | Tương quan với `Rings` |
|---|---:|
| `ShellWeight` | 0.6276 |
| `Diameter` | 0.5747 |
| `Height` | 0.5575 |
| `Length` | 0.5567 |
| `WholeWeight` | 0.5404 |
| `VisceraWeight` | 0.5038 |
| `ShuckedWeight` | 0.4209 |

Từ đây, nhóm rút ra hai điểm đáng chú ý:

- `ShellWeight` là biến có tương quan tuyến tính mạnh nhất với `Rings`, cho thấy khối lượng vỏ khô là một tín hiệu sinh học quan trọng cho bài toán tuổi
- các biến kích thước và khối lượng đều có tương quan dương mức trung bình đến khá, tức là càng lớn và càng nặng thì bào ngư có xu hướng có nhiều vòng hơn

Tuy nhiên, các hệ số tương quan chưa quá cao đến mức có thể giải thích hoàn toàn bài toán bằng mô hình tuyến tính đơn giản. Điều này cũng giải thích vì sao nhóm cần thử thêm các mô hình phi tuyến như Random Forest, Gradient Boosting, SVR và MLP.

## 2.5. Các quyết định tiền xử lý rút ra từ EDA

Từ các kết quả trên, notebook `01_kham_pha_du_lieu.ipynb` dẫn tới các quyết định kỹ thuật quan trọng cho các bước sau:

1. Biến `Sex` phải được mã hóa trước khi huấn luyện.
2. Không cần quy trình điền khuyết dữ liệu, nhưng vẫn cần bước kiểm tra missing value trong pipeline để đảm bảo tính tái lập.
3. Cần thử ít nhất hai hướng xử lý dữ liệu số:
   - giữ nguyên thang đo sau khi mã hóa cho nhóm mô hình cây
   - chuẩn hóa cho nhóm mô hình nhạy với scale
4. Do các biến khối lượng có độ lệch và outlier, cần thêm một nhánh xử lý `log1p + RobustScaler` để kiểm chứng giả thuyết về dữ liệu lệch phân phối.
5. Cần ưu tiên thử các mô hình có khả năng học quan hệ phi tuyến.

# CHƯƠNG 3: PHƯƠNG PHÁP VÀ QUÁ TRÌNH THỰC NGHIỆM

## 3.1. Quy trình triển khai tổng thể

Toàn bộ phần thực nghiệm của đồ án được chia thành 5 notebook chính, tương ứng với một quy trình nghiên cứu tuần tự:

1. `01_kham_pha_du_lieu.ipynb`: hiểu dữ liệu, phát hiện vấn đề và rút ra định hướng xử lý.
2. `02_tien_xu_ly_du_lieu.ipynb`: xây dựng các phiên bản dữ liệu sạch và phù hợp với từng nhóm mô hình.
3. `03_huan_luyen_mo_hinh.ipynb`: huấn luyện 10 mô hình baseline và đánh giá bằng `5-fold Cross-validation`.
4. `04_danh_gia_mo_hinh.ipynb`: kiểm tra khả năng tổng quát hóa trên tập test.
5. `05_thu_nghiem_bo_sung.ipynb`: tối ưu hóa, so sánh các phiên bản dữ liệu, tuning và thử ensemble.

Cách tổ chức này giúp báo cáo không chỉ trình bày kết quả cuối cùng, mà còn thể hiện được quá trình nghiên cứu, suy luận và kiểm chứng giả thuyết của nhóm qua từng giai đoạn.

## 3.2. Giai đoạn 1: Khám phá dữ liệu và hình thành giả thuyết

Ở notebook đầu tiên, mục tiêu của nhóm không phải là huấn luyện mô hình ngay, mà là trả lời các câu hỏi nền tảng:

- dữ liệu có sạch hay không
- biến nào có khả năng dự báo mạnh
- có tồn tại độ lệch phân phối hoặc ngoại lệ đáng chú ý hay không
- bài toán có xu hướng tuyến tính hay phi tuyến

Kết quả EDA đã tạo ra bộ giả thuyết nghiên cứu ban đầu:

- `Sex` là biến hữu ích nên phải được mã hóa
- các biến khối lượng mang tín hiệu mạnh và có thể tiếp tục khai thác bằng feature engineering
- quan hệ giữa đầu vào và `Rings` không hoàn toàn tuyến tính
- cần thử song song mô hình tuyến tính, mô hình kernel, mô hình cây và mạng nơ-ron

Nhờ đó, notebook EDA đóng vai trò như nền tảng ra quyết định cho toàn bộ các giai đoạn sau, thay vì chỉ là bước minh họa biểu đồ.

## 3.3. Giai đoạn 2: Tiền xử lý dữ liệu

Notebook `02_tien_xu_ly_du_lieu.ipynb` hiện thực hóa các kết luận từ EDA thành pipeline xử lý dữ liệu cụ thể.

### 3.3.1. Làm sạch dữ liệu

Nhóm sử dụng các hàm trong `src/data/clean_data.py` để:

- kiểm tra lại missing values
- kiểm tra và loại bỏ duplicate nếu có
- chuẩn hóa định dạng cột phân loại

Sau bước này, dữ liệu vẫn giữ nguyên kích thước `4177 x 9`, xác nhận rằng bộ dữ liệu đầu vào đã khá sạch ngay từ đầu.

### 3.3.2. Chia dữ liệu train/test

Nhóm chia dữ liệu theo tỷ lệ `70/30` với `random_state = 42`, thu được:

- `X_train`: `(2923, 8)`
- `X_test`: `(1254, 8)`
- `y_train`: `(2923,)`
- `y_test`: `(1254,)`

Việc cố định seed giúp kết quả thực nghiệm có thể tái lập.

### 3.3.3. Tạo ba phiên bản dữ liệu

Một điểm quan trọng trong notebook 02 là nhóm không ép mọi mô hình dùng chung một phiên bản dữ liệu, mà thiết kế ba nhánh xử lý khác nhau:

| Phiên bản | Cách xử lý | Mục đích sử dụng chính |
|---|---|---|
| `encoded_only` | One-Hot `Sex`, giữ nguyên biến số | Phù hợp cho mô hình cây |
| `standard_scaled` | One-Hot `Sex` + `StandardScaler` cho biến số | Phù hợp cho mô hình nhạy scale |
| `robust_log_scaled` | `log1p` cho nhóm khối lượng + One-Hot + `RobustScaler` | Kiểm tra giả thuyết về outlier và độ lệch |

Sau khi mã hóa One-Hot cho `Sex`, cả ba phiên bản đều có `10` cột đầu vào.

### 3.3.4. Kiểm soát rò rỉ dữ liệu

Một điểm thể hiện sự cẩn thận trong triển khai là mọi encoder và scaler đều chỉ được `fit` trên tập train, sau đó mới `transform` sang tập test. Quy trình này tránh hiện tượng data leakage, bảo đảm rằng mô hình không vô tình nhìn thấy thông tin của tập kiểm tra ở bước tiền xử lý.

### 3.3.5. Lưu dữ liệu trung gian

Các tập dữ liệu được lưu vào `data/interim` và `data/processed` để các notebook sau có thể dùng lại trực tiếp. Cách làm này có ba lợi ích:

- giảm lặp lại thao tác xử lý
- giúp các bước huấn luyện và đánh giá chạy độc lập hơn
- hỗ trợ việc tái hiện toàn bộ quy trình khi cần báo cáo hoặc kiểm tra lại

## 3.4. Giai đoạn 3: Huấn luyện 10 mô hình baseline

Notebook `03_huan_luyen_mo_hinh.ipynb` là giai đoạn trọng tâm đầu tiên để lượng hóa hiệu năng của các thuật toán.

### 3.4.1. Mười mô hình hồi quy được sử dụng

Nhóm triển khai đủ 10 mô hình hồi quy bắt buộc:

| STT | Mô hình | Phiên bản dữ liệu dùng cho baseline |
|---|---|---|
| 1 | `LinearRegression` | `standard_scaled` |
| 2 | `KNeighborsRegressor` | `standard_scaled` |
| 3 | `RidgeRegression` | `standard_scaled` |
| 4 | `DecisionTreeRegressor` | `encoded_only` |
| 5 | `RandomForestRegressor` | `encoded_only` |
| 6 | `GradientBoostingRegressor` | `encoded_only` |
| 7 | `SGDRegressor` | `standard_scaled` |
| 8 | `SVR` | `standard_scaled` |
| 9 | `LinearSVR` | `standard_scaled` |
| 10 | `MLPRegressor` | `standard_scaled` |

Việc gán phiên bản dữ liệu theo bản chất thuật toán là một điểm mạnh của đồ án. Nếu tất cả mô hình bị ép dùng chung một kiểu preprocess, kết quả so sánh sẽ dễ trở nên thiếu công bằng.

### 3.4.2. Phương pháp đánh giá

Tất cả mô hình baseline đều được đánh giá bằng:

- `5-fold Cross-validation`
- `KFold(n_splits=5, shuffle=True, random_state=42)`
- các metric: `MSE`, `RMSE`, `RSE`, `MAE`, `execution_time_sec`

Ngoài các metric tổng hợp, notebook còn lưu dự đoán OOF để phục vụ các bước so sánh sâu hơn ở giai đoạn tối ưu hóa.

### 3.4.3. Kết quả baseline trên cross-validation

| Mô hình | Dataset | RMSE | MAE | MSE | RSE | Thời gian (s) |
|---|---|---:|---:|---:|---:|---:|
| `GradientBoostingRegressor` | `encoded_only` | 2.1717 | 1.5304 | 4.7163 | 0.4497 | 1.3256 |
| `SVR` | `standard_scaled` | 2.1806 | 1.4823 | 4.7549 | 0.4534 | 1.4044 |
| `MLPRegressor` | `standard_scaled` | 2.2146 | 1.4979 | 4.9047 | 0.4677 | 7.4329 |
| `RandomForestRegressor` | `encoded_only` | 2.2198 | 1.5640 | 4.9276 | 0.4699 | 2.3422 |
| `LinearRegression` | `standard_scaled` | 2.2622 | 1.6026 | 5.1175 | 0.4880 | 5.2685 |
| `RidgeRegression` | `standard_scaled` | 2.2625 | 1.6020 | 5.1189 | 0.4881 | 0.0295 |
| `SGDRegressor` | `standard_scaled` | 2.2901 | 1.6114 | 5.2445 | 0.5001 | 0.0493 |
| `LinearSVR` | `standard_scaled` | 2.2913 | 1.5672 | 5.2501 | 0.5007 | 0.0466 |
| `KNeighborsRegressor` | `standard_scaled` | 2.3393 | 1.6213 | 5.4723 | 0.5218 | 3.4916 |
| `DecisionTreeRegressor` | `encoded_only` | 3.0191 | 2.0859 | 9.1150 | 0.8692 | 0.1312 |

### 3.4.4. Nhận xét từ baseline CV

Từ bảng trên, nhóm rút ra một số nhận xét quan trọng:

- `GradientBoostingRegressor` là mô hình baseline tốt nhất theo `RMSE`, `MSE` và `RSE`.
- `SVR` cho `MAE` thấp nhất, chứng tỏ mô hình này kiểm soát sai số tuyệt đối khá tốt.
- `MLPRegressor` chưa đứng đầu trong CV, nhưng nằm trong nhóm rất cạnh tranh.
- `DecisionTreeRegressor` kém xa phần còn lại, cho thấy cây đơn lẻ không đủ mạnh cho bài toán này.
- `LinearRegression` và `RidgeRegression` có kết quả gần như tương đương, phản ánh rằng regularization tuyến tính chỉ cải thiện rất ít khi chưa thay đổi không gian đặc trưng.

Kết quả baseline cũng củng cố kết luận từ EDA: bài toán AbaloneAge có xu hướng cần mô hình phi tuyến hoặc mô hình có năng lực biểu diễn mạnh hơn.

## 3.5. Giai đoạn 4: Đánh giá khả năng tổng quát hóa trên tập test

Notebook `04_danh_gia_mo_hinh.ipynb` dùng các mô hình đã huấn luyện trên toàn bộ tập train để dự đoán trên tập test và kiểm tra khả năng tổng quát hóa thực sự.

### 3.5.1. Kết quả baseline trên tập test

| Mô hình | Dataset | RMSE test | MAE test | MSE test | RSE test |
|---|---|---:|---:|---:|---:|
| `MLPRegressor` | `standard_scaled` | 2.0970 | 1.4871 | 4.3976 | 0.4331 |
| `SVR` | `standard_scaled` | 2.1609 | 1.4914 | 4.6694 | 0.4598 |
| `GradientBoostingRegressor` | `encoded_only` | 2.1800 | 1.5400 | 4.7523 | 0.4680 |
| `RidgeRegression` | `standard_scaled` | 2.1853 | 1.5829 | 4.7757 | 0.4703 |
| `LinearRegression` | `standard_scaled` | 2.1874 | 1.5832 | 4.7848 | 0.4712 |
| `SGDRegressor` | `standard_scaled` | 2.1934 | 1.5855 | 4.8111 | 0.4738 |
| `RandomForestRegressor` | `encoded_only` | 2.1946 | 1.5433 | 4.8165 | 0.4743 |
| `LinearSVR` | `standard_scaled` | 2.2074 | 1.5363 | 4.8724 | 0.4798 |
| `KNeighborsRegressor` | `standard_scaled` | 2.2463 | 1.5914 | 5.0457 | 0.4969 |
| `DecisionTreeRegressor` | `encoded_only` | 2.9863 | 2.0678 | 8.9179 | 0.8782 |

### 3.5.2. Phát hiện quan trọng từ bước đánh giá

Kết quả test cho thấy một điểm rất đáng chú ý: mô hình tốt nhất trên tập test không phải `GradientBoostingRegressor`, mà là `MLPRegressor` với `RMSE_test = 2.0970`.

Điều này cho thấy:

- xếp hạng trên cross-validation và xếp hạng trên test không hoàn toàn trùng nhau
- `MLPRegressor` có khả năng tổng quát hóa tốt hơn so với mức thể hiện trên CV
- việc chọn mô hình cuối cùng không thể chỉ dựa vào kết quả cross-validation

Mặt khác, `SVR` vẫn giữ được vị trí rất cao ở cả CV và test, cho thấy đây là một mô hình ổn định. `GradientBoostingRegressor` tuy dẫn đầu ở CV nhưng khi ra test lại xếp sau `MLPRegressor` và `SVR`.

### 3.5.3. So sánh chênh lệch giữa CV và test

Một quan sát thú vị từ notebook 04 là phần lớn mô hình có `RMSE_test` nhỏ hơn `RMSE_CV`. Ví dụ:

- `MLPRegressor`: `2.2146 -> 2.0970`
- `SVR`: `2.1806 -> 2.1609`
- `RandomForestRegressor`: `2.2198 -> 2.1946`

Riêng `GradientBoostingRegressor` có xu hướng gần như ổn định, với chênh lệch rất nhỏ:

- `2.1717 -> 2.1800`

Từ đây có thể nhận xét rằng:

- chưa xuất hiện dấu hiệu overfitting nghiêm trọng trên các baseline mạnh nhất
- `GradientBoostingRegressor` là mô hình ổn định
- `MLPRegressor` là mô hình có khả năng bứt lên tốt trên tập test trong cấu hình baseline hiện tại

### 3.5.4. Góc nhìn về chi phí tính toán

Notebook 04 còn phân tích thêm thời gian huấn luyện và suy luận:

- nhanh nhất theo thời gian huấn luyện/CV+fit: `RidgeRegression` (`0.0295s`), `LinearSVR` (`0.0466s`)
- nhanh nhất theo inference trên test: `LinearRegression` (`0.0009s`), `SGDRegressor` (`0.0013s`)
- chậm nhất ở inference: `SVR` (`0.3976s`)
- `MLPRegressor` có độ chính xác cao nhất nhưng thời gian huấn luyện lớn (`7.4329s`)

Điều này gợi ý rằng nếu chỉ ưu tiên tốc độ, nhóm có thể cân nhắc mô hình tuyến tính hoặc SGD. Tuy nhiên, nếu ưu tiên độ chính xác, `MLPRegressor` và `SVR` là hai lựa chọn nổi bật hơn.

## 3.6. Giai đoạn 5: Thử nghiệm bổ sung và tối ưu hóa

Notebook `05_thu_nghiem_bo_sung.ipynb` là phần thể hiện rõ nhất tinh thần nghiên cứu của đồ án, bởi nhóm không dừng lại ở việc báo cáo baseline mà tiếp tục kiểm tra ít nhất 5 hướng cải thiện khác nhau.

### 3.6.1. So sánh `standard_scaled` và `robust_log_scaled`

Đầu tiên, nhóm kiểm tra giả thuyết xuất phát từ EDA: nếu các biến khối lượng có độ lệch và outlier, liệu `robust_log_scaled` có giúp các mô hình nhạy với scale hoạt động tốt hơn hay không.

Kết quả cho thấy:

| Mô hình | Phiên bản tốt hơn |
|---|---|
| `KNeighborsRegressor` | `standard_scaled` |
| `LinearRegression` | `robust_log_scaled` |
| `LinearSVR` | `robust_log_scaled` |
| `MLPRegressor` | `robust_log_scaled` |
| `RidgeRegression` | `robust_log_scaled` |
| `SGDRegressor` | `robust_log_scaled` |
| `SVR` | `standard_scaled` |

Như vậy, `robust_log_scaled` không phải là lời giải tốt nhất cho mọi mô hình, nhưng lại giúp cải thiện đáng kể cho phần lớn các mô hình tuyến tính và mạng nơ-ron. Đây là một kết quả rất quan trọng, vì nó cho thấy việc tiền xử lý nên gắn với từng thuật toán cụ thể, thay vì chọn một cấu hình duy nhất cho toàn bộ pipeline.

### 3.6.2. Tuning siêu tham số cho toàn bộ 10 mô hình

Sau khi xác định phiên bản dữ liệu phù hợp hơn cho từng mô hình, nhóm sử dụng `RandomizedSearchCV` để tuning cả 10 thuật toán. Một số không gian tìm kiếm điển hình bao gồm:

- `SVR`: `C`, `gamma`, `epsilon`
- `RandomForestRegressor`: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- `GradientBoostingRegressor`: `n_estimators`, `learning_rate`, `subsample`, `max_depth`
- `MLPRegressor`: kích thước hidden layer, `alpha`, `learning_rate_init`

Cách làm này thể hiện rõ nỗ lực nghiên cứu của nhóm: không chỉ tối ưu cho 1-2 mô hình mạnh nhất, mà thực hiện tuning có hệ thống cho toàn bộ bộ 10 mô hình bắt buộc.

### 3.6.3. Kết quả optimized single models

Kết quả sau tuning trên cross-validation:

| Mô hình | Dataset | RMSE CV | MAE CV |
|---|---|---:|---:|
| `MLPRegressor` | `robust_log_scaled` | 2.1168 | 1.4768 |
| `SVR` | `standard_scaled` | 2.1516 | 1.4804 |
| `RandomForestRegressor` | `encoded_only` | 2.1681 | 1.5261 |
| `GradientBoostingRegressor` | `encoded_only` | 2.1708 | 1.5210 |
| `LinearRegression` | `robust_log_scaled` | 2.1843 | 1.5684 |
| `RidgeRegression` | `robust_log_scaled` | 2.1843 | 1.5684 |
| `LinearSVR` | `robust_log_scaled` | 2.1851 | 1.5691 |
| `SGDRegressor` | `robust_log_scaled` | 2.1884 | 1.5671 |
| `KNeighborsRegressor` | `standard_scaled` | 2.2770 | 1.5715 |
| `DecisionTreeRegressor` | `encoded_only` | 2.4455 | 1.7406 |

Kết quả optimized trên tập test:

| Mô hình | Dataset | RMSE test | MAE test |
|---|---|---:|---:|
| `SVR` | `standard_scaled` | 2.1422 | 1.4923 |
| `MLPRegressor` | `robust_log_scaled` | 2.1468 | 1.5499 |
| `RandomForestRegressor` | `encoded_only` | 2.1513 | 1.5307 |
| `GradientBoostingRegressor` | `encoded_only` | 2.1669 | 1.5179 |
| `SGDRegressor` | `robust_log_scaled` | 2.1687 | 1.5574 |
| `LinearSVR` | `robust_log_scaled` | 2.1753 | 1.5606 |
| `RidgeRegression` | `robust_log_scaled` | 2.1770 | 1.5597 |
| `LinearRegression` | `robust_log_scaled` | 2.1770 | 1.5597 |
| `KNeighborsRegressor` | `standard_scaled` | 2.1941 | 1.5186 |
| `DecisionTreeRegressor` | `encoded_only` | 2.4366 | 1.7726 |

### 3.6.4. So sánh baseline và optimized

Nhìn vào kết quả optimized, nhóm rút ra những điểm rất đáng giá về mặt nghiên cứu:

- `SVR` cải thiện từ `RMSE_test = 2.1609` xuống `2.1422`
- `RandomForestRegressor` cải thiện từ `2.1946` xuống `2.1513`
- `GradientBoostingRegressor` cải thiện từ `2.1800` xuống `2.1669`
- `LinearRegression` và `RidgeRegression` cũng cải thiện nhẹ khi chuyển sang `robust_log_scaled`

Tuy nhiên, phát hiện quan trọng hơn là:

- mô hình optimized tốt nhất trên test là `SVR` với `2.1422`
- nhưng mô hình baseline tốt nhất vẫn là `MLPRegressor` với `2.0970`

Nói cách khác, tối ưu hóa đã cải thiện nhiều mô hình riêng lẻ so với baseline của chính chúng, nhưng vẫn chưa vượt qua được baseline mạnh nhất toàn cục. Đây là một kết quả rất có ý nghĩa học thuật, vì nó minh họa rõ ràng rằng:

- tuning không phải lúc nào cũng dẫn tới mô hình cuối cùng tốt nhất
- kết quả CV tốt hơn không đảm bảo rằng kết quả test cũng tốt hơn
- một baseline phù hợp đôi khi mạnh hơn cả các mô hình đã tối ưu hóa nếu dữ liệu và cấu hình ban đầu đã tương đối hợp lý

### 3.6.5. Ensemble: Weighted Average và Stacking

Sau bước tuning, nhóm tiếp tục thử hai chiến lược kết hợp mô hình:

1. `WeightedAverageEnsemble` từ top 3 mô hình optimized
2. `StackingRidge` với meta-model `Ridge`

Ba mô hình được dùng làm thành viên chính là:

- `MLPRegressor`
- `SVR`
- `RandomForestRegressor`

Kết quả ensemble:

| Mô hình | Thành viên | RMSE CV | RMSE test | MAE test |
|---|---|---:|---:|---:|
| `WeightedAverageEnsemble` | `MLP + SVR + RF` | 2.1052 | 2.1105 | 1.4924 |
| `StackingRidge` | `MLP + SVR + RF` | 2.0977 | 2.1280 | 1.5404 |

Nhận xét:

- `WeightedAverageEnsemble` cho kết quả test khá tốt, đứng trên hầu hết các mô hình optimized đơn lẻ
- `StackingRidge` cho CV rất mạnh (`2.0977`) nhưng khi ra test lại tăng lên `2.1280`
- cả hai ensemble vẫn chưa vượt được baseline `MLPRegressor` (`2.0970`)

Điều này cho thấy ensemble có thể cải thiện độ mạnh trung bình của hệ thống, nhưng trong bài toán hiện tại, lợi ích đó vẫn chưa đủ để đánh bại mô hình baseline mạnh nhất.

## 3.7. Tổng kết quá trình thực nghiệm

Nhìn toàn bộ 5 notebook như một chuỗi nghiên cứu, nhóm đã đạt được các kết quả quan trọng sau:

- làm sạch và mô tả được bộ dữ liệu Abalone một cách có hệ thống
- thiết kế được ba phiên bản dữ liệu phù hợp với các nhóm mô hình khác nhau
- huấn luyện đủ 10 mô hình hồi quy baseline bằng `5-fold Cross-validation`
- đánh giá được khả năng tổng quát hóa trên tập test
- kiểm tra có hệ thống 5 hướng cải thiện khác nhau ở giai đoạn tối ưu hóa
- chứng minh bằng số liệu rằng không phải kỹ thuật nâng cao nào cũng vượt được baseline tốt nhất

# CHƯƠNG 4: KẾT QUẢ VÀ THẢO LUẬN

## 4.1. Thành quả nổi bật nhất của đồ án

Nếu chỉ nhìn riêng từng giai đoạn, có thể thấy nhiều mô hình khác nhau nổi bật ở những tiêu chí khác nhau. Tuy nhiên, nếu xét mục tiêu cuối cùng là chọn mô hình có khả năng tổng quát hóa tốt trên tập test, kết luận rõ ràng nhất của đồ án là:

- `MLPRegressor (standard_scaled)` là baseline tốt nhất với `RMSE_test = 2.0970`
- `SVR` là optimized single model tốt nhất với `RMSE_test = 2.1422`
- `WeightedAverageEnsemble` là ensemble tốt nhất với `RMSE_test = 2.1105`

Như vậy, mô hình cuối cùng được ưu tiên lựa chọn ở trạng thái hiện tại vẫn là baseline `MLPRegressor`.

## 4.2. Vì sao preprocessing không tạo ra bước nhảy lớn

Các kết quả từ notebook 02 và notebook 05 cho thấy thay đổi preprocessing có ảnh hưởng, nhưng không phải là yếu tố quyết định tuyệt đối. Có thể giải thích điều này như sau:

- bộ dữ liệu Abalone vốn đã khá sạch, không có missing value và không có nhiễu nghiêm trọng
- số lượng đặc trưng chỉ ở mức vừa phải, nên việc chuẩn hóa không tạo ra khác biệt quá lớn như ở các bộ dữ liệu rất cao chiều
- các mô hình cây như Random Forest và Gradient Boosting gần như không phụ thuộc vào scaling
- với một số mô hình, `robust_log_scaled` giúp ích; nhưng với `SVR` và `KNN`, `standard_scaled` lại vẫn là lựa chọn tốt hơn

Điều này dẫn đến một bài học quan trọng: tiền xử lý nên được xem là một lựa chọn theo từng mô hình, chứ không nên coi là một “phép màu” giúp cải thiện đồng loạt.

## 4.3. Vì sao tuning chưa vượt được baseline mạnh nhất

Đây là phần thảo luận quan trọng nhất của báo cáo. Tại sao sau khi tuning toàn bộ 10 mô hình, mô hình tốt nhất trên test vẫn không vượt được baseline `MLPRegressor`?

Có thể đưa ra một số lý giải hợp lý:

- baseline `MLPRegressor` ban đầu vốn đã phù hợp tốt với cấu trúc dữ liệu sau `standard_scaled`
- quá trình tuning tối ưu theo cross-validation có thể khiến mô hình trở nên phù hợp hơn với train folds nhưng không tăng tương ứng trên test
- các khoảng cải thiện thu được ở nhiều mô hình là có thật, nhưng chủ yếu ở quy mô nhỏ
- ensemble giúp ổn định dự đoán, nhưng khi các mô hình thành viên đã có lỗi gần nhau, mức cộng hưởng không đủ lớn để tạo bứt phá rõ rệt

Kết quả này không phải là “thất bại” của tuning, mà trái lại là một thành quả nghiên cứu quan trọng. Nó cho thấy nhóm đã kiểm chứng được một vấn đề rất thường gặp trong học máy thực nghiệm: **cải thiện trên CV không đồng nghĩa với cải thiện trên dữ liệu chưa thấy**.

## 4.4. So sánh theo từng nhóm mô hình

### 4.4.1. Nhóm mô hình tuyến tính

`LinearRegression`, `RidgeRegression` và `SGDRegressor` cho kết quả ở mức trung bình khá. Chúng không đứng đầu, nhưng vẫn duy trì sai số tương đối ổn định quanh vùng `RMSE ≈ 2.17 - 2.29` sau tối ưu hóa. Điều này cho thấy:

- bài toán AbaloneAge có một phần quan hệ gần tuyến tính
- nhưng quan hệ này chưa đủ để mô hình tuyến tính vượt lên nhóm kernel, boosting hay neural network

### 4.4.2. Nhóm mô hình cây

`RandomForestRegressor` và `GradientBoostingRegressor` đều hoạt động tốt. Đặc biệt:

- `GradientBoostingRegressor` tốt nhất ở baseline CV
- `RandomForestRegressor` cải thiện đáng kể sau tuning

Điều này xác nhận rằng các quan hệ phi tuyến giữa kích thước, khối lượng và số vòng tuổi là có thật, và mô hình cây khai thác được phần lớn thông tin đó.

### 4.4.3. Nhóm mô hình kernel

`SVR` là một trong những mô hình ổn định nhất toàn bộ đồ án:

- đứng thứ 2 ở baseline CV
- đứng thứ 2 ở baseline test
- là optimized single model tốt nhất ở test

Tính ổn định này cho thấy `SVR` phù hợp tốt với dữ liệu Abalone, đặc biệt khi dùng phiên bản `standard_scaled`.

### 4.4.4. Nhóm mạng nơ-ron

`MLPRegressor` là mô hình đáng chú ý nhất của báo cáo:

- không phải mạnh nhất ở baseline CV
- nhưng là mạnh nhất ở baseline test
- sau tuning, CV cải thiện mạnh (`2.2146 -> 2.1168`) nhưng test lại không vượt baseline ban đầu

Điều này cho thấy `MLPRegressor` rất nhạy với cấu hình và cách tiền xử lý. Nó có tiềm năng lớn, nhưng cũng dễ thay đổi hành vi khi chuyển sang cấu hình tối ưu hóa khác.

## 4.5. Giá trị học thuật của các kết quả âm

Trong nhiều đồ án, sinh viên thường chỉ trình bày các bước “có cải thiện”. Tuy nhiên, điểm mạnh của chuỗi notebook trong repo này là nhóm đã ghi lại cả những thử nghiệm không đem lại kết quả tốt nhất. Về mặt nghiên cứu, đây là điều rất có giá trị vì:

- nó giúp chứng minh rằng nhóm đã thử nhiều hướng chứ không chỉ chọn lọc kết quả đẹp
- nó cho thấy nhóm có khả năng phân tích vì sao một hướng không hiệu quả
- nó làm báo cáo thuyết phục hơn, vì người đọc thấy được toàn bộ quá trình ra quyết định

Ví dụ rõ nhất là:

- `StackingRidge` có CV rất đẹp nhưng test không thắng baseline
- tuning nhiều mô hình giúp cải thiện cục bộ nhưng không đánh bại `MLPRegressor`
- preprocessing nâng cao chỉ hữu ích với một số mô hình nhất định

Những “kết quả âm” này thực chất chính là bằng chứng nghiên cứu rất quan trọng.

## 4.6. Bổ sung: ý nghĩa của hướng feature engineering trong repo

Bên cạnh 5 notebook chính, repo còn có thí nghiệm bổ sung về feature engineering trong các file kết quả thực nghiệm riêng. Kết quả này rất đáng chú ý vì nó bổ sung thêm một hướng cải thiện khác với tuning thuần túy.

Trong thí nghiệm này, nhóm so sánh hai cấu hình:

- `raw_features`
- `manual_feature_engineering`

Ba đặc trưng mới được bổ sung là:

- `VolumeApprox`
- `TotalWeightToShell`
- `EdibleWeightRatio`

Kết quả cho thấy:

- `RMSE` giảm từ `2.1844` xuống `2.1542`
- `MAE` giảm từ `1.5457` xuống `1.5268`
- `R2` tăng từ `0.5301` lên `0.5430`

Điều này củng cố thêm một kết luận rất quan trọng của đồ án: trong bài toán AbaloneAge, việc thiết kế đặc trưng có ý nghĩa sinh học và hình học có thể đem lại cải thiện rõ rệt, thậm chí đáng kể hơn so với chỉ thay đổi kỹ thuật chuẩn hóa dữ liệu.

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. Kết luận chung

Thông qua 5 notebook thực nghiệm, đồ án đã xây dựng được một quy trình nghiên cứu tương đối đầy đủ cho bài toán dự đoán số vòng tuổi của bào ngư. Quy trình này không chỉ dừng lại ở việc chạy mô hình, mà đã đi từ:

- hiểu và mô tả bản chất dữ liệu
- lựa chọn chiến lược tiền xử lý theo từng nhóm mô hình
- huấn luyện baseline có hệ thống
- đánh giá tổng quát hóa trên tập test
- thực hiện các thử nghiệm cải thiện, tuning và ensemble

Về mặt kết quả, có thể tổng kết như sau:

- `MLPRegressor (standard_scaled)` là mô hình baseline tốt nhất trên tập test với `RMSE = 2.0970`
- `SVR` là mô hình optimized đơn lẻ tốt nhất với `RMSE = 2.1422`
- `WeightedAverageEnsemble` là ensemble tốt nhất với `RMSE = 2.1105`
- `GradientBoostingRegressor` là mô hình baseline tốt nhất trên cross-validation

Từ đó, kết luận thực nghiệm quan trọng nhất của đồ án là: **mô hình baseline phù hợp đôi khi có thể hiệu quả hơn cả các mô hình đã tối ưu hóa sâu**, và vì vậy việc đánh giá trên tập test phải luôn được xem là bước quyết định cuối cùng.

## 5.2. Những bài học rút ra từ quá trình nghiên cứu

Qua chuỗi thực nghiệm, nhóm rút ra một số bài học có giá trị:

1. Dữ liệu sạch và tổ chức pipeline đúng có thể tạo ra baseline rất mạnh.
2. Không nên giả định rằng preprocessing phức tạp hơn sẽ luôn tốt hơn.
3. Tuning cần được đánh giá đồng thời trên CV và test, không nên chỉ nhìn vào một phía.
4. Ensemble không tự động tốt hơn mô hình đơn lẻ mạnh nhất.
5. Cách trình bày quá trình thử nghiệm theo từng giả thuyết giúp báo cáo thuyết phục và có chiều sâu hơn hẳn cách chỉ liệt kê thuật toán.

## 5.3. Hướng phát triển tiếp theo

Từ những gì đã đạt được, nhóm có thể tiếp tục mở rộng đồ án theo các hướng sau:

- bổ sung thí nghiệm feature engineering sâu hơn, đặc biệt các đặc trưng có ý nghĩa hình học và sinh học như thể tích xấp xỉ, tỷ lệ phần thịt và tỷ lệ giữa các loại khối lượng
- thử các mô hình boosting hiện đại hơn hoặc thư viện tối ưu hơn nếu phạm vi đề tài cho phép
- phân tích kỹ hơn residual theo từng vùng tuổi để xem mô hình sai nhiều ở nhóm bào ngư trẻ hay già
- áp dụng thêm phương pháp chọn đặc trưng để giảm dư thừa giữa các biến kích thước và khối lượng
- đánh giá độ ổn định bằng nhiều lần random split hoặc nested cross-validation để có kết luận chắc chắn hơn

Tổng thể, đồ án đã đạt được mục tiêu chính về mặt học thuật và thực hành: xây dựng được một pipeline học máy tương đối hoàn chỉnh, có khả năng tái lập, có so sánh định lượng rõ ràng và có phần thảo luận phản ánh đúng bản chất của quá trình nghiên cứu thực nghiệm.
