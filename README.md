# Nhận dạng hoạt động người đa phương thức trên MMAct

## Tóm tắt

Tài liệu này trình bày hệ thống nhận dạng hoạt động người (Human Activity
Recognition, HAR) sử dụng ba loại dữ liệu: video RGB, chuỗi khung xương
(skeleton) và tín hiệu quán tính IMU. Các mô hình đơn modality được huấn luyện
riêng, sau đó dự đoán được căn chỉnh theo `clip_id` và kết hợp bằng voting
xác suất hoặc attention-based fusion. Dự án cũng có các công cụ phân tích dữ
liệu (EDA), xử lý dữ liệu, huấn luyện, benchmark và nhận dạng liên tục.

Trong thí nghiệm fusion cuối, các dự đoán OOF trên train được dùng để fit
trọng số/attention và tập test được dùng để đánh giá. Trên 1.619 mẫu có đủ
dự đoán của cả ba modality, kết quả
cao nhất theo accuracy là attention-self với 0,9660; kết quả cao nhất theo
macro-F1 là hard voting ba modality với 0,9626. Đây là các giá trị quan sát
trên test để báo cáo đầy đủ, không phải căn cứ để chọn phương pháp tốt nhất.

> **Phạm vi số liệu:** Các bảng dưới đây phân biệt EDA trên manifest train,
> benchmark mô hình modality trên validation và benchmark fusion trên test.
> Chúng dùng các tập mẫu/protocol khác nhau và không nên so sánh trực tiếp
> nếu không xét đến protocol. README giữ nguyên các liên kết tới sơ đồ kiến
> trúc sẵn có trong dự án.

## 1. Giới thiệu

Nhận dạng hoạt động từ một modality đơn có thể gặp giới hạn do che khuất,
nhiễu quan sát hoặc tín hiệu không phân biệt được một số lớp. Dự án khảo sát
khả năng bổ trợ giữa:

- **RGB:** thông tin ngoại quan và bối cảnh cảnh quay.
- **Skeleton:** cấu trúc khớp và chuyển động cơ thể.
- **IMU:** tín hiệu gia tốc đeo trên cơ thể.

Mục tiêu là xây dựng pipeline thực nghiệm từ manifest dữ liệu, EDA, huấn
luyện mô hình từng modality đến fusion có kiểm soát rò rỉ. Hệ thống cũng tổ
chức các thành phần inference theo cửa sổ thời gian và hậu xử lý chuỗi dự
đoán; tuy nhiên, số liệu continuous không nằm trong thí nghiệm fusion được
báo cáo ở đây.

## 2. Kiến trúc hệ thống

Sơ đồ tổng quan, sơ đồ thu thập dữ liệu, sơ đồ kiến trúc huấn luyện và sơ đồ
realtime bên dưới được giữ nguyên từ README trước.

![Kiến trúc tổng thể](<docs/Kiến trúc tổng quan.png>)

![Collect Data](<docs/Collect Data.png>)

![Train Model Architecture](<docs/Train Model Architecture.png>)

![Realtime Architecture](<docs/Realtime Architecture.png>)

Về mặt xử lý, pipeline thực nghiệm được tổ chức như sau:

```text
MMAct manifests + raw modality files
                 │
                 ├── EDA: phân phối lớp, độ dài chuỗi, chất lượng dữ liệu
                 │
                 ├── Skeleton model ──┐
                 ├── IMU model ────────┼── căn chỉnh clip_id ── voting/attention
                 └── RGB model ────────┘                         │
                                                                 └── test metrics
```

## 3. Dữ liệu và thiết lập chia tập

### 3.1 Các lớp hoạt động

Thí nghiệm dùng 35 lớp: 32 lớp hoạt động MMAct và ba lớp mở rộng
`carrying_heavy`, `carrying_light`, `setting_down`.

### 3.2 Manifest multimodal

Manifest multimodal được tạo bằng cách gộp manifest train/validation nguồn,
giữ các clip có đủ file RGB, skeleton và IMU, sau đó chia thành train,
validation và test theo nhóm sự kiện vật lý
`subject/scene/session/action`. Audit ghi nhận:

| Split | Số dòng manifest | Số lớp | Sự kiện vật lý trùng với split khác |
|---|---:|---:|---:|
| Train | 11.182 | 35 | 0 |
| Validation | 3.727 | 35 | 0 |
| Test | 3.727 | 35 | 0 |

Audit xác nhận không có `clip_id` trùng và không có sự kiện vật lý trùng giữa
các split. File audit cũng cho thấy cả ba split có camera `cam1`, `cam2`,
`cam3`; do đó split hiện tại đảm bảo tách sự kiện vật lý, **không phải** đánh
giá holdout camera độc lập.

Các con số trong manifest không đồng nghĩa mỗi modality có cùng số bản ghi
độc lập. Dataset IMU loại trùng theo sự kiện vật lý vì một sự kiện có thể
xuất hiện ở nhiều góc camera. Bước fusion tiếp tục căn chỉnh theo giao
`clip_id` giữa các đầu ra: tập test fusion còn 1.619 mẫu; tập fit OOF có
4.861 mẫu. Đây là lý do số mẫu fusion nhỏ hơn số dòng manifest.

### 3.3 Phân phối lớp

Trong train manifest 11.182 dòng, EDA ghi nhận đủ 35 lớp. Số lượng lớp thấp
nhất là 135 và cao nhất là 512 mẫu; trung bình khoảng 319,5 mẫu/lớp. Một số
lớp ít mẫu gồm `drinking` (135), `using_pc` (135), `pocket_in` (136),
`pocket_out` (136), `sitting` (136), `sitting_down` (136) và `standing_up`
(136). Các lớp `talking_on_phone` (512) và `using_phone` (510) có nhiều mẫu
hơn. Phân phối vì vậy chưa cân bằng hoàn toàn; macro-F1 được báo cáo cùng
accuracy để phản ánh hiệu quả trên các lớp.

Biểu đồ EDA về phân phối lớp ba modality được tạo cùng các thống kê EDA.

## 4. Phân tích dữ liệu khám phá (EDA)

EDA được chạy trên manifest train, kiểm tra mẫu file và lưu thống kê
cùng hình. Số lượng 11.182 ở bảng dưới là số dòng manifest
được phân tích; phần kiểm tra chất lượng thực hiện trên một mẫu con theo cấu
hình của từng EDA module.

| Modality | Dòng manifest | Chi tiết độ dài/metadata | Kiểm tra chất lượng |
|---|---:|---|---|
| Skeleton | 11.182 | 300 clip được kiểm tra; độ dài 52–1.262 frame, trung vị 183, trung bình 214,2 | 200/200 mẫu hợp lệ (100%); không thiếu file, sai shape, NaN hoặc dữ liệu toàn 0 |
| IMU | 11.182 | 500 chuỗi được lấy mẫu; độ dài 112–2.104 điểm, trung vị 318, trung bình 373,1; tần số giả định 50 Hz | 300/300 mẫu hợp lệ (100%); không thiếu file, sai shape, NaN hoặc quá ngắn |
| RGB | 11.182 | 200 video được lấy mẫu; 1920×1080, khoảng 29,97 FPS; thời lượng trung bình 7,63 giây (trung vị 6,72 giây) | 100/100 mẫu mở và hợp lệ (100%); 200/200 video lấy metadata thành công |

Các thống kê trên mô tả **mẫu EDA**, không khẳng định đã kiểm tra toàn bộ
11.182 file. Với IMU, thống kê kênh được tính trên một mẫu 500 chuỗi. Ví dụ
gia tốc có các kênh `ax`, `ay`, `az`; trong cấu hình huấn luyện IMU hiện tại,
feature set `gravity_jerk` mở rộng chúng thành 10 kênh dẫn xuất.

EDA tạo thống kê JSON, biểu đồ phân phối lớp, độ dài chuỗi, chất lượng và
mẫu dữ liệu cho từng modality; đồng thời tạo biểu đồ phân phối lớp tổng hợp.
Các quan sát chính theo modality:

- **Skeleton:** 17 khớp COCO; trong 300 chuỗi lấy mẫu, độ dài trung bình
  214,2 frame, trung vị 183 frame, khoảng tứ phân vị 138–249 frame. Độ dài
  biến thiên đáng kể, vì vậy mô hình cần xử lý chuỗi có thời lượng khác nhau;
  pipeline resample chuỗi về độ dài cố định trước khi đưa vào mô hình.
- **IMU:** trong 500 chuỗi, độ dài trung bình là 373 mẫu và trung vị 318
  mẫu. Với tần số mục tiêu 50 Hz, thời lượng trung bình xấp xỉ 7,65 giây.
  Các kênh gia tốc thô có scale/phân phối khác nhau, tạo động lực chuẩn hóa
  trên thống kê train và thêm đặc trưng gravity/jerk.
- **RGB:** các video được lấy mẫu có độ phân giải 1920×1080, tốc độ khung
  hình trung bình 29,97 FPS; thời lượng từ khoảng 2,54 đến 23,12 giây. Vì
  số frame và thời lượng thay đổi, quy trình lấy mẫu frame tạo đầu vào có
  kích thước thời gian cố định cho mô hình.
- **Chất lượng:** các mẫu được kiểm tra không cho thấy file thiếu, shape
  lỗi hoặc NaN theo các tiêu chí module. Đây là kiểm tra trên mẫu con, không
  thay thế việc kiểm định toàn bộ dữ liệu.

Phân phối nhãn có mất cân bằng vừa phải: lớp lớn nhất có 512 mẫu, lớp nhỏ
nhất có 135 mẫu, tỷ lệ khoảng 3,79 lần. Vì vậy, accuracy được xem cùng
macro-F1; macro-F1 gán trọng số ngang nhau cho từng lớp và nhạy hơn với
những lớp ít mẫu.

### 4.1 Phân phối lớp trên ba modality

Biểu đồ dưới đây so sánh số clip theo lớp trên manifest train. Do IMU được
deduplicate theo sự kiện vật lý khi xây dựng dataset, biểu đồ phản ánh số
dòng trong manifest chung, không phải số mẫu độc lập cuối cùng mà mỗi
dataset loader đưa vào huấn luyện.

![Phân phối lớp trên RGB, Skeleton và IMU](<outputs/eda/combined/class_balance_3mod.png>)

### 4.2 Skeleton

Phân phối lớp Skeleton kế thừa cấu trúc mất cân bằng đã mô tả. Độ dài chuỗi
thay đổi lớn giữa các clip; thống kê vị trí và độ tin cậy khớp giúp kiểm tra
chất lượng keypoint, còn chuyển động trung bình theo lớp cho thấy mức độ
động học khác nhau giữa các hoạt động.

![Phân phối lớp Skeleton](<outputs/eda/skeleton/class_distribution.png>)

![Độ dài chuỗi Skeleton](<outputs/eda/skeleton/sequence_length.png>)

![Thống kê tọa độ và độ tin cậy theo khớp](<outputs/eda/skeleton/joint_stats_heatmap.png>)

![Năng lượng chuyển động Skeleton theo lớp](<outputs/eda/skeleton/motion_energy_per_class.png>)

![Ví dụ chuỗi Skeleton](<outputs/eda/skeleton/sample_skeletons.png>)

### 4.3 IMU

Độ dài tín hiệu IMU phân tán quanh trung vị 318 mẫu. Phân phối từng kênh
gia tốc và dạng sóng đại diện theo lớp được minh họa bên dưới; đây là tín
hiệu thô ba trục, trước khi biến đổi thành feature set 10 kênh
`gravity_jerk` sử dụng khi huấn luyện.

![Phân phối lớp IMU](<outputs/eda/imu/class_distribution.png>)

![Độ dài chuỗi IMU](<outputs/eda/imu/sequence_length.png>)

![Phân phối giá trị các kênh gia tốc IMU](<outputs/eda/imu/channel_stats.png>)

![Tín hiệu IMU đại diện theo lớp](<outputs/eda/imu/signals_per_class.png>)

![Thời lượng tín hiệu IMU](<outputs/eda/imu/duration_distribution.png>)

### 4.4 RGB

Video có tốc độ khung hình gần 30 FPS và cùng độ phân giải 1920×1080 trong
mẫu khảo sát. Hình metadata mô tả độ dài/FPS/kích thước video; hình frame
mẫu cung cấp kiểm tra trực quan về nội dung RGB được đưa vào pipeline.

![Phân phối lớp RGB](<outputs/eda/rgb/class_distribution.png>)

![Metadata video RGB](<outputs/eda/rgb/video_metadata.png>)

![Các frame RGB mẫu](<outputs/eda/rgb/sample_frames.png>)

## 5. Tiền xử lý và biểu diễn đầu vào

### 5.1 Skeleton

Skeleton sử dụng 17 keypoint COCO. Chuỗi được căn chỉnh theo root/hip và
chuẩn hóa theo khoảng cách hai vai theo cấu hình; chuỗi thời gian được resample
về 64 frame. Các stream có thể chứa tọa độ khớp cùng thông tin bone/motion,
tùy kiến trúc được benchmark.

### 5.2 IMU

IMU dùng gia tốc watch, resample theo cấu hình 50 Hz. Feature set `gravity_jerk`
gồm gia tốc XYZ, độ lớn gia tốc, thành phần động sau tách gravity và jerk,
tổng cộng 10 kênh. Cấu hình cửa sổ mô hình là 250 điểm; chuẩn hóa dùng
thống kê train.

### 5.3 RGB

RGB được đọc từ video, lấy mẫu frame và chuẩn hóa theo cấu hình data. Các
kiến trúc video được benchmark trên clip RGB đã tiền xử lý.

## 6. Benchmark mô hình đơn modality

Các kết quả dưới đây là benchmark validation cho từng nhánh dữ liệu, không
phải kết quả fusion test. Ở mỗi nhánh, kiến trúc được chọn cho fusion hiện
tại được đánh dấu rõ và so sánh với các mô hình còn lại.

### 6.1 Nhánh Skeleton — CTR-GCN Head

Skeleton benchmark gồm 3.727 mẫu validation. Bảng báo cáo đầy đủ các metric
chất lượng và chi phí huấn luyện đã được ghi nhận. Số tham số tính theo
triệu (M); thời gian huấn luyện tính theo giây.

| Kiến trúc Skeleton | Accuracy val | Macro-F1 val | Weighted-F1 val | Tham số (M) | Train (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| dual | **0,9077** | **0,9070** | **0,9089** | 3,39 | 1.628,0 | 43 |
| **ctrgcn_head** | 0,9042 | 0,9031 | 0,9049 | 2,49 | 1.608,7 | 70 |
| ctrgcn | 0,8983 | 0,8986 | 0,8990 | 1,43 | 1.482,6 | 65 |
| stgcn | 0,8854 | 0,8870 | 0,8864 | 3,07 | 500,7 | 61 |
| sta_gcn | 0,8806 | 0,8814 | 0,8820 | 0,69 | 272,4 | 79 |
| ms_g3d | 0,8723 | 0,8701 | 0,8733 | 0,73 | 329,2 | 74 |

`dual` dẫn đầu cả ba metric chất lượng. So với `dual`, `ctrgcn_head` giảm
Macro-F1 0,0039 nhưng dùng ít hơn 0,90M tham số (khoảng 26,5%); thời gian
huấn luyện ghi nhận cũng thấp hơn nhẹ. So với backbone `ctrgcn`, head này
cải thiện Macro-F1 0,0045 với mức tăng 1,06M tham số. Đây là một trade-off
hợp lý giữa chất lượng và độ phức tạp, nhưng không phải lựa chọn dẫn đầu
theo metric. Nó được dùng trong fusion run hiện tại; bảng không chứng minh
rằng nó là mô hình Skeleton tối ưu.

### 6.2 Nhánh RGB — MC3-18

RGB benchmark gồm 3.500 mẫu validation. `mc3_18` đứng đầu về accuracy và
Weighted-F1; `r2plus1d_18` đứng đầu về Macro-F1.

| Kiến trúc RGB | Accuracy val | Macro-F1 val | Weighted-F1 val | Tham số (M) | Train (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| **mc3_18** | **0,8071** | 0,7996 | **0,7884** | 11,51 | 13.398,9 | 29 |
| r2plus1d_18 | 0,8026 | **0,8022** | 0,7810 | 31,32 | 10.032,3 | 8 |
| mvit_v2_s | 0,7991 | 0,7957 | 0,7801 | 34,26 | 45.470,9 | 20 |
| r3d_18 | 0,7826 | 0,7799 | 0,7664 | 33,18 | 11.596,5 | 30 |
| s3d | 0,7437 | 0,7382 | 0,7256 | 7,95 | 4.480,6 | 9 |
| videomae2 | 0,6840 | 0,6818 | 0,6683 | 21,90 | 10.923,6 | 20 |
| videomae | 0,6769 | 0,6668 | 0,6549 | 86,26 | 11.320,1 | 9 |

Lý do định lượng để giữ `mc3_18` là điểm validation toàn diện: accuracy cao
nhất, Weighted-F1 cao nhất và Macro-F1 chỉ thấp hơn `r2plus1d_18` 0,0026.
Đổi lại, `mc3_18` có 11,51M tham số, thấp hơn 19,81M (khoảng 63,3%) so với
`r2plus1d_18`. Tuy nhiên, thời gian huấn luyện ghi nhận của MC3-18 dài hơn;
vì vậy, không nên diễn giải đây là mô hình nhanh nhất. Nếu ưu tiên Macro-F1
thay vì accuracy/Weighted-F1 và kích thước mô hình, `r2plus1d_18` là lựa
chọn cạnh tranh hơn.

### 6.3 Nhánh IMU — TCN

IMU benchmark có 1.622 mẫu validation sau khi loại trùng bản ghi theo sự
kiện vật lý. `fckd` dẫn đầu về accuracy và Macro-F1; `inceptiontime` dẫn
đầu về Weighted-F1.

| Kiến trúc IMU | Accuracy val | Macro-F1 val | Weighted-F1 val | Tham số (M) | Train (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| fckd | **0,7879** | **0,7878** | 0,7874 | 0,30 | 102,0 | 88 |
| inceptiontime | 0,7793 | 0,7805 | **0,7814** | 0,48 | 72,5 | 48 |
| transformer | 0,7793 | 0,7771 | 0,7795 | 0,40 | 102,2 | 71 |
| **tcn** | 0,7774 | 0,7765 | 0,7767 | 0,21 | 51,2 | 39 |
| cnn_bilstm_cbam | 0,7762 | 0,7725 | 0,7770 | 0,80 | 116,2 | 87 |
| cnn_bilstm | 0,7583 | 0,7539 | 0,7573 | 0,80 | 74,0 | 48 |
| xtinyhar | 0,6535 | 0,6419 | 0,6470 | 0,09 | 54,7 | 45 |

TCN là phương án gọn và có thời gian huấn luyện thấp nhất trong các mô hình
IMU được liệt kê: 0,21M tham số và 51,2 giây. So với `fckd`, TCN giảm khoảng
30% số tham số và giảm một nửa thời gian huấn luyện, nhưng Macro-F1 thấp hơn
0,0113. Vì vậy, TCN có lý do lựa chọn khi ưu tiên cấu hình gọn/chi phí chạy;
nếu ưu tiên metric validation, `fckd` là ứng viên tốt hơn. TCN là kiến trúc
được dùng trong fusion run hiện tại, không phải mô hình dẫn đầu benchmark.

### 6.4 Tổng hợp và tiêu chí lựa chọn cho fusion

| Nhánh | Kiến trúc dùng trong fusion | Accuracy val | Macro-F1 val | Weighted-F1 val | Tham số (M) | Lý do giữ trong cấu hình hiện tại |
|---|---|---:|---:|---:|---:|---|
| Skeleton | `ctrgcn_head` | 0,9042 | 0,9031 | 0,9049 | 2,49 | Gần điểm `dual`, ít tham số hơn; không phải mô hình có F1 cao nhất |
| RGB | `mc3_18` | 0,8071 | 0,7996 | 0,7884 | 11,51 | Accuracy/Weighted-F1 cao nhất; Macro-F1 gần nhất với `r2plus1d_18`, ít tham số hơn |
| IMU | `tcn` | 0,7774 | 0,7765 | 0,7767 | 0,21 | Mô hình gọn, thời gian huấn luyện thấp; đánh đổi một phần điểm F1 so với `fckd` |

**Diễn giải lựa chọn.** Bộ `ctrgcn_head + mc3_18 + tcn` được báo cáo vì đây
là các checkpoint đã tạo OOF và dùng trong fusion run hiện tại. Các bảng
đưa ra lý do hậu nghiệm về trade-off chất lượng–độ phức tạp, nhưng không có
biên bản cho thấy một quy tắc chọn đa mục tiêu đã được định trước và áp
dụng đồng nhất cho cả ba nhánh. Vì vậy, không nên tuyên bố rằng đây là bộ
kiến trúc tối ưu theo một tiêu chí duy nhất. Nếu chọn lại cho một thí nghiệm
mới, cần định trước metric ưu tiên (ví dụ Macro-F1), giới hạn chi phí và
chọn mô hình trên validation/meta-validation độc lập trước khi đánh giá test.

Các cột chi phí là số tham số và thời gian huấn luyện; benchmark không ghi
nhận FLOPs, throughput hay latency suy luận, nên không thể kết luận trực
tiếp về tốc độ triển khai thực tế. Thời gian huấn luyện còn phụ thuộc môi
trường/phần cứng và cấu hình chạy. Các số mẫu validation khác nhau
(Skeleton 3.727, RGB 3.500, IMU 1.622), nên chỉ dùng từng bảng để xếp hạng
mô hình trong cùng nhánh; không so sánh trực tiếp điểm giữa các modality.
Một số kiến trúc là biến thể/ablation của dự án, không phải tái lập hoàn
toàn bài báo gốc.

## 7. Phương pháp fusion và giao thức OOF

Sau khi benchmark và cố định các model đơn modality, xác suất đầu ra được
căn chỉnh theo cùng `clip_id`. Các thí nghiệm gồm ba tổ hợp đôi và một tổ hợp
ba modality; đồng thời chạy từng modality đơn làm baseline.

### 7.1 Voting

- **Hard voting:** mỗi model bỏ một phiếu cho lớp có xác suất lớn nhất.
  Nhãn có số phiếu nhiều nhất là dự đoán tổng hợp.
- **Average probability:** lấy trung bình đều các vector xác suất của các
  modality rồi chọn lớp có xác suất tổng hợp cao nhất.
- **Weighted probability:** tính tổng xác suất có trọng số
  \(P(c)=\sum_m w_m P_m(c)\). Tổng trọng số bằng 1. Lưới trọng số được tìm
  trên tập fit, tối ưu macro-F1; trọng số thu được sau đó được giữ cố định
  khi đánh giá tập test.

### 7.2 Attention-based fusion

Attention nhận vector xác suất từ các modality làm đầu vào và học cách phối
hợp chúng. Bốn biến thể được khảo sát:

- `self`: self-attention trên các modality.
- `cross`: cross-attention giữa các modality.
- `adaptive`: attention với điều chỉnh phụ thuộc đầu vào.
- `gate_cross`: cơ chế gating kết hợp cross-attention.

Meta-fuser được huấn luyện trên dự đoán OOF train. Một phần dữ liệu train
OOF được dành cho early stopping; phần dự đoán test chỉ được dùng sau khi
fit để tạo kết quả cuối. `fit_epoch` và `fit_f1_macro` trong bảng fusion là
thông tin lựa chọn epoch/dừng của meta-fuser trên phần train dành cho dừng,
không phải điểm test.

### 7.3 Vì sao dùng OOF

Nếu học trọng số hoặc attention trên dự đoán in-sample, base model đã thấy
những clip đó khi huấn luyện; xác suất thường quá lạc quan và meta-fuser có
thể học từ dự đoán không đại diện cho dữ liệu mới. OOF (out-of-fold) hạn chế
vấn đề này:

1. Chia train thành 5 fold theo nhóm sự kiện vật lý.
2. Với mỗi fold, huấn luyện base model trên các fold còn lại.
3. Dự đoán fold bị giữ lại; mỗi clip vì vậy được dự đoán bởi model chưa train
   trên chính clip đó.
4. Ghép các dự đoán OOF theo `clip_id` và dùng chúng để fit trọng số voting
   hoặc attention.
5. Đóng băng lựa chọn từ bước fit và đánh giá trên test chưa tham gia quá
   trình fit.

Trong lần chạy được trình bày, giao OOF đủ ba modality gồm 4.861 clip và
giao test gồm 1.619 clip. Số clip khác nhau là do các modality có độ phủ dữ
liệu khác nhau và các dự đoán phải được căn chỉnh trên giao `clip_id`.
OOF bảo vệ bước fit fusion khỏi dự đoán train in-sample; nó không tự chọn
kiến trúc base model thay cho validation benchmark.

## 8. Kết quả fusion trên test

### 8.1 Giao thức đánh giá

Run fusion sử dụng:

| Thành phần | Thiết lập |
|---|---|
| Split fit | `train_cross_view` |
| Nguồn dự đoán fit | OOF |
| Số mẫu giao fit | 4.861 |
| Split đánh giá | `test_cross_view` |
| Số mẫu giao test | 1.619 |
| Kiến trúc | skeleton `ctrgcn_head`, IMU `tcn`, RGB `mc3_18` |
| Fusion methods | voting và attention (`both`) |

OOF nghĩa là mỗi dự đoán train được tạo bởi mô hình mà fold huấn luyện tương
ứng không chứa mẫu đó. Trọng số voting và attention được fit từ dự đoán OOF,
còn test được giữ lại để đánh giá. Kết quả được ghi theo từng phương pháp;
không chọn một “best” dựa trên test.

### 8.2 Kết quả từng modality trên giao test

| Thí nghiệm | Modality | Phương pháp | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---:|---:|---:|
| E1 | Skeleton | Single | 0,9271 | 0,9247 | 0,9283 |
| E2 | IMU | Single | 0,7801 | 0,7737 | 0,7803 |
| E3 | RGB | Single | 0,8660 | 0,8789 | 0,8679 |

Ở giao test hiện tại, IMU đơn modality thấp hơn skeleton và RGB. Đây là điểm
cần được thảo luận như một hạn chế/quan sát thực nghiệm; không nên thay bằng
kết quả từ một benchmark khác dùng split hoặc số mẫu khác.

### 8.3 Kết quả tổ hợp hai modality

| Thí nghiệm | Method | Accuracy | Macro-F1 | Weighted-F1 | Trọng số voting |
|---|---|---:|---:|---:|---|
| E4: Skeleton + IMU | Hard | 0,9531 | 0,9503 | 0,9539 | 0,5 / 0,5 |
| E4: Skeleton + IMU | Average | 0,9543 | 0,9513 | 0,9551 | 0,5 / 0,5 |
| E4: Skeleton + IMU | Weighted | 0,9543 | 0,9513 | 0,9551 | 0,5 / 0,5 |
| E4: Skeleton + IMU | Attention-self | 0,9450 | 0,9366 | 0,9429 | — |
| E4: Skeleton + IMU | Attention-cross | 0,9382 | 0,9243 | 0,9343 | — |
| E4: Skeleton + IMU | Attention-adaptive | 0,9364 | 0,9264 | 0,9337 | — |
| E4: Skeleton + IMU | Attention-gate_cross | 0,9481 | 0,9376 | 0,9451 | — |
| E5: Skeleton + RGB | Hard | 0,9500 | 0,9476 | 0,9504 | 0,5 / 0,5 |
| E5: Skeleton + RGB | Average | 0,9500 | 0,9476 | 0,9504 | 0,5 / 0,5 |
| E5: Skeleton + RGB | Weighted | 0,9500 | 0,9476 | 0,9504 | 0,5 / 0,5 |
| E5: Skeleton + RGB | Attention-self | 0,9599 | 0,9570 | 0,9594 | — |
| E5: Skeleton + RGB | Attention-cross | 0,9500 | 0,9390 | 0,9451 | — |
| E5: Skeleton + RGB | Attention-adaptive | 0,9568 | 0,9496 | 0,9543 | — |
| E5: Skeleton + RGB | Attention-gate_cross | 0,9561 | 0,9471 | 0,9523 | — |
| E6: IMU + RGB | Hard | 0,9438 | 0,9429 | 0,9437 | 0,5 / 0,5 |
| E6: IMU + RGB | Average | 0,9456 | 0,9445 | 0,9455 | 0,5 / 0,5 |
| E6: IMU + RGB | Weighted | 0,9456 | 0,9445 | 0,9455 | 0,5 / 0,5 |
| E6: IMU + RGB | Attention-self | 0,9271 | 0,9244 | 0,9257 | — |
| E6: IMU + RGB | Attention-cross | 0,9327 | 0,9287 | 0,9304 | — |
| E6: IMU + RGB | Attention-adaptive | 0,9382 | 0,9304 | 0,9345 | — |
| E6: IMU + RGB | Attention-gate_cross | 0,9327 | 0,9285 | 0,9296 | — |

### 8.4 Kết quả tổ hợp ba modality

| Thí nghiệm | Method | Accuracy | Macro-F1 | Weighted-F1 | Trọng số voting |
|---|---|---:|---:|---:|---|
| E7: Skeleton + IMU + RGB | Hard | 0,9642 | **0,9626** | 0,9646 | 1/3 mỗi modality |
| E7: Skeleton + IMU + RGB | Average | 0,9648 | 0,9617 | **0,9651** | 1/3 mỗi modality |
| E7: Skeleton + IMU + RGB | Weighted | 0,9629 | 0,9592 | 0,9634 | SK 0,3; IMU 0,4; RGB 0,3 |
| E7: Skeleton + IMU + RGB | Attention-self | **0,9660** | 0,9587 | 0,9636 | — |
| E7: Skeleton + IMU + RGB | Attention-cross | 0,9629 | 0,9580 | 0,9620 | — |
| E7: Skeleton + IMU + RGB | Attention-adaptive | 0,9605 | 0,9554 | 0,9598 | — |
| E7: Skeleton + IMU + RGB | Attention-gate_cross | 0,9648 | 0,9562 | 0,9621 | — |

Các giá trị in đậm chỉ đánh dấu metric lớn nhất trong nhóm E7 để giúp đọc
bảng; không hàm ý chọn checkpoint/phương pháp cuối cùng bằng cách tối ưu trên
test. Khi chọn một phương pháp để triển khai hoặc báo cáo như cấu hình chính,
cần căn cứ validation/meta-validation độc lập. Vì kết quả test này đã được
quan sát, việc dùng nó để quyết định cấu hình sẽ khiến test không còn là đánh
giá hoàn toàn độc lập.

### 8.5 Diễn giải

1. Kết hợp modality thường cải thiện rõ so với từng modality đơn trong thí
   nghiệm này. Ví dụ E7 hard voting đạt macro-F1 0,9626, so với 0,9247 của
   skeleton đơn, 0,7737 của IMU đơn và 0,8789 của RGB đơn trên cùng giao test.
2. Không có một phương pháp fusion thắng mọi metric. E7 attention-self có
   accuracy cao nhất (0,9660), trong khi hard voting có macro-F1 cao nhất
   (0,9626) và average có weighted-F1 cao nhất (0,9651).
3. Weighted voting E7 chọn trọng số gần nhau, hơi tăng trọng số IMU
   (0,3/0,4/0,3), nhưng không đạt điểm cao hơn hard/average trên test. Trọng
   số này được fit trên OOF, không được tối ưu bằng test.
4. Attention không nhất thiết vượt voting: kết quả thay đổi tùy tổ hợp
   modality. Điều này gợi ý cần đánh giá thêm độ ổn định qua nhiều seed hoặc
   tập kiểm tra độc lập.
5. Tập đánh giá fusion là giao clip có dự đoán đủ modality, không đại diện
   cho toàn bộ 3.727 dòng test manifest; việc loại trùng IMU làm giảm quy mô
   xuống 1.619.

## 9. Hạn chế và nguy cơ diễn giải

- **IMU có điểm số thấp hơn trong fusion:** macro-F1 IMU đơn là 0,7737 trên
  giao test. Cần kiểm tra thêm preprocessing, đồng bộ, nhãn sự kiện, đặc
  trưng và sự khác biệt phân phối giữa các split trước khi kết luận nguyên
  nhân.
- **Số mẫu đa modality bị giới hạn bởi giao dữ liệu:** fusion sử dụng 1.619
  clip test. Chỉ báo cáo điểm trên tập giao chung và nêu rõ số mẫu.
- **Camera không được holdout:** audit cho thấy các camera xuất hiện ở mọi
  split, dù sự kiện vật lý không trùng. Không nên mô tả kết quả hiện tại là
  khả năng tổng quát hóa sang camera chưa thấy.
- **Lựa chọn phương pháp:** test đã được chạy và kết quả đã xem. Không dùng
  điểm test để chọn hyperparameter hoặc phương pháp rồi xem lại chính test
  như một đánh giá khách quan độc lập.
- **Continuous recognition:** các thành phần inference/postprocess hiện có
  trong mã nguồn, nhưng README này không gán số liệu continuous nếu không có
  artifact evaluation tương ứng.
- **Tái lập paper:** một số backbone hoặc tên kiến trúc là adaptation/biến
  thể của dự án. Xem provenance trong benchmark CSV trước khi mô tả là tái
  lập chính thức.

## 10. Tái lập thí nghiệm

### 10.1 Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cài PyTorch đúng phiên bản/CUDA của máy nếu môi trường hiện tại chưa nhận GPU.

### 10.2 Tạo split multimodal (nếu cần tạo lại)

```bash
python scripts/04_split_multimodal_train_val_test.py \
  --input-dir data/processed/MMAct/manifests \
  --output-dir data/processed/MMAct/manifests/multimodal_cross_view_3way \
  --split cross_view --seed 42
```

Script kiểm tra file tồn tại, đủ modality, đủ 35 lớp và không trùng sự kiện
giữa split. Không ghi đè lên manifest nguồn.

### 10.3 Chạy EDA

```bash
python scripts/05_run_eda.py \
  --manifest-dir data/processed/MMAct/manifests/multimodal_cross_view_3way \
  --output-root outputs/eda
```

### 10.4 Huấn luyện modality

Ví dụ train kiến trúc đang dùng cho fusion:

```bash
python scripts/13_benchmark_models.py \
  --modality skeleton --mode train --split cross_view \
  --architectures ctrgcn_head \
  --manifest-dir data/processed/MMAct/manifests/multimodal_cross_view_3way

python scripts/13_benchmark_models.py \
  --modality imu --mode train --split cross_view \
  --architectures tcn \
  --manifest-dir data/processed/MMAct/manifests/multimodal_cross_view_3way

python scripts/13_benchmark_models.py \
  --modality rgb --mode train --split cross_view \
  --architectures mc3_18 \
  --manifest-dir data/processed/MMAct/manifests/multimodal_cross_view_3way
```

### 10.5 Sinh OOF và chạy fusion

Thay các đường dẫn run mẫu bằng đúng checkpoint muốn đánh giá:

```bash
python scripts/15_generate_oof_predictions.py \
  --manifest-dir data/processed/MMAct/manifests/multimodal_cross_view_3way \
  --skeleton-run-dir outputs/runs/skeleton/<selected-run> \
  --imu-run-dir outputs/runs/imu/<selected-run> \
  --rgb-run-dir outputs/runs/rgb/<selected-run>

python scripts/16_benchmark_fusion.py \
  --config configs/fusion/voting.yaml \
  --fusion-mode both \
  --tag <experiment-tag>
```

Artifact lần chạy gồm bảng kết quả, protocol, cấu hình, danh sách clip được
đánh giá và log thực thi.

## 11. Cấu trúc mã nguồn và artifact

```text
configs/                 Cấu hình lớp, dữ liệu, train, fusion
data/processed/MMAct/    Dữ liệu đã xử lý và manifest
outputs/eda/             EDA JSON, biểu đồ và tóm tắt
outputs/benchmark/       Benchmark mô hình theo modality
outputs/runs/            Checkpoint, metrics, dự đoán và log từng lần chạy
scripts/                 Entry points xử lý, train, EDA, fusion, evaluation
src/data/                Dataset và preprocessing
src/models/              Mô hình Skeleton, IMU, RGB
src/fusion/              Căn chỉnh dự đoán, voting, weights, attention
src/inference/           Sliding-window và continuous inference
src/postprocess/         Smoothing, linking, removing và segment pipeline
```

## 12. Kết luận

Pipeline đã tạo được EDA cho ba modality, benchmark mô hình đơn modality và
thí nghiệm fusion đủ bảy tổ hợp (ba đơn modality, ba cặp modality và một tổ
hợp ba modality). Ở thí nghiệm fusion hiện tại, ba modality với hard voting
đạt macro-F1 0,9626 và attention-self đạt accuracy 0,9660 trên 1.619 mẫu test
giao chung. OOF được dùng cho bước fit fusion, còn test được dùng để đánh giá.

Kết quả là bằng chứng thực nghiệm của pipeline trên bộ dữ liệu và giao thức
được mô tả, không phải khẳng định hiệu năng tổng quát cho mọi camera, người
dùng hoặc bối cảnh triển khai. Các bước tiếp theo nên ưu tiên chọn cấu hình
trên validation, giữ một test độc lập chưa xem để báo cáo cuối, kiểm tra kỹ
IMU, đánh giá qua nhiều seed và thực hiện continuous evaluation trên dữ liệu
liên tục có ground truth.
