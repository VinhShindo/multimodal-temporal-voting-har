# Nhận dạng hoạt động người đa phương thức trên MMAct

## Tóm tắt

Tài liệu này trình bày hệ thống nhận dạng hoạt động người (Human Activity
Recognition, HAR) sử dụng ba loại dữ liệu: video RGB, chuỗi khung xương
(skeleton) và tín hiệu quán tính IMU. Các mô hình đơn modality được huấn luyện
riêng, sau đó dự đoán được căn chỉnh theo `clip_id` và kết hợp bằng voting
xác suất hoặc attention-based fusion. Dự án cũng có các công cụ phân tích dữ
liệu (EDA), xử lý dữ liệu, huấn luyện, benchmark và nhận dạng liên tục.

> **Trạng thái protocol:** Quy trình hiện tại tạo sáu protocol chỉ gồm
> `train`/`val`, mỗi modality có manifest riêng. Các số liệu benchmark cũ
> trong tài liệu này thuộc split legacy; chúng không phải kết quả của sáu
> protocol mới và không dùng để so sánh protocol. Upload completeness chưa
> được xác nhận; audit của manifest snapshot hiện là `INCOMPLETE`, nên EDA mới
> phản ánh các file đang có và chưa phải thống kê cuối cùng.

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
                                                                 └── validation metrics
```

## 3. Dữ liệu và thiết lập chia tập

### 3.1 Các lớp hoạt động

Thí nghiệm dùng 35 lớp: 32 lớp hoạt động MMAct và ba lớp mở rộng
`carrying_heavy`, `carrying_light`, `setting_down`.

### 3.2 Sáu protocol train/val

| Protocol | Cách chia |
|---|---|
| `x_scene` | Giữ nguyên train/val từ split nguồn cross-scene |
| `x_view` | Giữ nguyên train/val từ split nguồn cross-view |
| `x_sub_scene` | 80/20 theo group `(subject, scene)` |
| `x_sub_view` | 80/20 theo group `(subject, cam)` |
| `x_session_scene` | 80/20 theo group `(subject, scene, session)` |
| `x_session_view` | 80/20 theo group `(subject, cam, session)` |

Mỗi protocol ghi manifest riêng cho RGB, skeleton và IMU; không yêu cầu một
clip phải có đủ cả ba modality. Với IMU, nhiều camera có thể trỏ tới cùng
physical event: event được dedupe, val được ưu tiên khi event xuất hiện ở cả
hai phía, và IMU manifest không có cột camera. Split nguồn cross-scene được
giữ nguyên như hiện có; không giả định scene ID của nguồn hoàn toàn disjoint.

### 3.3 Phân phối lớp

EDA mới chạy trên toàn bộ train+val của `x_sub_scene`, được tạo từ hợp nhất
hai split nguồn trimmed. Có 27.527 clip RGB/Skeleton và 8.679 sự kiện IMU
sau khi deduplicate. Cả ba modality có đủ 35 lớp. RGB/Skeleton có 356–1.257
mẫu mỗi lớp; IMU có 100–388 sự kiện mỗi lớp. Các phân phối chưa cân bằng
hoàn toàn, vì vậy macro-F1 được báo cáo cùng accuracy.

Biểu đồ phân phối lớp ba modality và thống kê JSON được lưu trong
`outputs/eda/`.

### 3.4 Cấu trúc dữ liệu gốc

Dữ liệu hiện có được chia thành `data/raw/trimmed/` (clip RGB, pose, IMU và
split x-scene/x-view) và `data/raw/untrimmed/` (video, sensor, annotation theo
session cùng split x-session). Kiểm kê đường dẫn và số lượng file hiện có được
ghi tại [data/raw/README.md](./data/raw/README.md); các số đếm này mô tả snapshot
đang có, không khẳng định mọi tổ hợp subject/session đã đầy đủ.

## 4. Phân tích dữ liệu khám phá (EDA)

EDA được chạy trên **toàn bộ train+val rows** của `x_sub_scene`; class
distribution và thống kê độ dài manifest dùng toàn bộ các dòng. Các phép
đọc dữ liệu nhị phân, metadata video và quality check vẫn dùng mẫu con theo
giới hạn của từng EDA module.

| Modality | Dòng manifest | Chi tiết độ dài/metadata | Kiểm tra chất lượng mẫu |
|---|---:|---|---|
| Skeleton | 27.527 | Độ dài 22–1.544 frame; trung vị 183, trung bình 213,3; 300 chuỗi dùng cho joint stats | 196/200 hợp lệ (98%); 4 mẫu toàn 0, không thiếu file, sai shape hoặc NaN |
| IMU | 8.679 | 500 chuỗi: 109–1.372 điểm, trung vị 312, trung bình 351,6; giả định 50 Hz | 300/300 hợp lệ (100%), không thiếu file, sai shape, NaN hoặc quá ngắn |
| RGB | 27.527 | 200 video metadata thành công; 1920×1080, FPS trung bình 29,968; thời lượng trung bình 6,95 giây (trung vị 5,74 giây) | 100/100 video mở và có frame (100%) |

IMU có ít dòng hơn RGB/Skeleton do event-level deduplication và các dòng
không có đường dẫn IMU; mọi đường dẫn IMU có trong manifest đều tồn tại.
Manifest audit được giữ ở trạng thái `INCOMPLETE` vì lần tạo này không khẳng
định upload đã hoàn tất. Các thống kê quality là kết quả trên mẫu, không phải
bảo đảm chất lượng của toàn bộ file.

EDA tạo JSON, biểu đồ phân phối lớp, độ dài chuỗi, chất lượng và mẫu trực
quan cho từng modality, cùng biểu đồ lớp tổng hợp. Các quan sát chính:

- **Skeleton:** 17 khớp COCO; phân vị độ dài manifest là 139–249 frame.
  Trong 300 chuỗi khảo sát, tọa độ chuẩn hóa có range khoảng -603 đến 503;
  4/200 mẫu quality check có toàn bộ tọa độ bằng 0, cần được kiểm tra thêm.
- **IMU:** thống kê kênh dùng 500 chuỗi ba trục `ax`, `ay`, `az`; độ dài trung
  bình 351,6 và trung vị 312 mẫu. Thời lượng 6,83 giây trung bình là ước tính
  theo giả định 50 Hz; feature set `gravity_jerk` tạo 10 kênh khi huấn luyện.
- **RGB:** các video được lấy mẫu có 1920×1080, FPS 29,95–29,97 và thời lượng
  1,77–22,32 giây; số frame thay đổi giữa các clip.
- **Untrimmed:** được kiểm kê cấu trúc file riêng tại
  [data/raw/README.md](./data/raw/README.md); EDA clip hiện tại chưa phân tích
  annotation hoặc đặc trưng theo session của dữ liệu này.

### 4.1 Phân phối lớp trên ba modality

Biểu đồ dưới đây so sánh số clip theo lớp trên toàn bộ manifest train+val. Do IMU được
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

Độ dài tín hiệu IMU phân tán quanh trung vị 312 mẫu. Phân phối từng kênh
gia tốc và dạng sóng đại diện theo lớp được minh họa bên dưới; đây là tín
hiệu thô ba trục, trước khi biến đổi thành feature set 10 kênh
`gravity_jerk` sử dụng khi huấn luyện.

![Phân phối lớp IMU](<outputs/eda/imu/class_distribution.png>)

![Độ dài chuỗi IMU](<outputs/eda/imu/sequence_length.png>)

![Phân phối giá trị các kênh gia tốc IMU](<outputs/eda/imu/channel_stats.png>)

![Tín hiệu IMU đại diện theo lớp](<outputs/eda/imu/signals_per_class.png>)

![Thời lượng tín hiệu IMU](<outputs/eda/imu/duration_distribution.png>)

### 4.4 RGB

Video có tốc độ khung hình 29,95–29,97 FPS và cùng độ phân giải 1920×1080
trong mẫu khảo sát. Hình metadata mô tả độ dài/FPS/kích thước video; hình frame
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
| **ctrgcn_head** | **0,9042** | **0,9031** | **0,9049** | 2,49 | 1.608,7 | 70 |
| ctrgcn | 0,8983 | 0,8986 | 0,8990 | 1,43 | 1.482,6 | 65 |
| stgcn | 0,8854 | 0,8870 | 0,8864 | 3,07 | 500,7 | 61 |
| sta_gcn | 0,8806 | 0,8814 | 0,8820 | 0,69 | 272,4 | 79 |
| ms_g3d | 0,8723 | 0,8701 | 0,8733 | 0,73 | 329,2 | 74 |

Trong các kiến trúc Skeleton được giữ lại, `ctrgcn_head` có điểm cao nhất ở
cả ba metric. So với backbone `ctrgcn`, head này cải thiện Macro-F1 0,0045
với mức tăng 1,06M tham số. Nó được dùng trong fusion run hiện tại; bảng
không chứng minh rằng đây là mô hình Skeleton tối ưu.

### 6.2 Nhánh RGB — MC3-18

RGB benchmark gồm 3.500 mẫu validation. `mc3_18` đứng đầu trong các kiến trúc
được giữ lại về accuracy, Macro-F1 và Weighted-F1.

| Kiến trúc RGB | Accuracy val | Macro-F1 val | Weighted-F1 val | Tham số (M) | Train (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| **mc3_18** | **0,8071** | 0,7996 | **0,7884** | 11,51 | 13.398,9 | 29 |
| mvit_v2_s | 0,7991 | 0,7957 | 0,7801 | 34,26 | 45.470,9 | 20 |
| r3d_18 | 0,7826 | 0,7799 | 0,7664 | 33,18 | 11.596,5 | 30 |
| s3d | 0,7437 | 0,7382 | 0,7256 | 7,95 | 4.480,6 | 9 |
| videomae2 | 0,6840 | 0,6818 | 0,6683 | 21,90 | 10.923,6 | 20 |
| videomae | 0,6769 | 0,6668 | 0,6549 | 86,26 | 11.320,1 | 9 |

Lý do định lượng để giữ `mc3_18` là điểm validation toàn diện trên các kiến
trúc còn được hỗ trợ: đứng đầu cả ba metric với 11,51M tham số. Thời gian
huấn luyện ghi nhận vẫn dài hơn một số kiến trúc khác; vì vậy, không nên
diễn giải đây là mô hình nhanh nhất.

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
| Skeleton | `ctrgcn_head` | 0,9042 | 0,9031 | 0,9049 | 2,49 | Điểm cao nhất trong các kiến trúc được giữ lại |
| RGB | `mc3_18` | 0,8071 | 0,7996 | 0,7884 | 11,51 | Accuracy/Macro-F1/Weighted-F1 cao nhất trong các kiến trúc được giữ lại |
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

### 6.5 Kết quả benchmark trên sáu protocol

Bảng dưới đây tổng hợp các benchmark protocol hiện tại với seed 42. Điểm
trung bình là trung bình số học không trọng số qua sáu protocol; số tham số
được tính theo triệu (M). Mỗi protocol có audit leakage riêng đạt PASS, nhưng
điều này không thay thế xác nhận rằng toàn bộ dữ liệu thô đã được tải lên.

| Nhánh | Kiến trúc | Tham số (M) | Accuracy TB | Macro-F1 TB | Weighted-F1 TB |
|---|---|---:|---:|---:|---:|
| IMU | `tcn` | 0,21 | 0,7753 | **0,7729** | 0,7756 |
| IMU | `inceptiontime` | 0,48 | 0,7620 | 0,7603 | 0,7646 |
| IMU | `cnn_bilstm_cbam` | 0,80 | 0,7492 | 0,7463 | 0,7489 |
| IMU | `cnn_bilstm` | 0,80 | 0,7476 | 0,7453 | 0,7473 |
| IMU | `fckd` | 0,30 | 0,7446 | 0,7407 | 0,7441 |
| IMU | `transformer` | 0,40 | 0,7191 | 0,7115 | 0,7174 |
| IMU | `xtinyhar` | 0,09 | 0,6515 | 0,6391 | 0,6475 |
| Skeleton | `ctrgcn_head` | 2,49 | **0,8806** | **0,8739** | 0,8802 |
| Skeleton | `ctrgcn` | 1,43 | 0,8655 | 0,8604 | 0,8655 |
| Skeleton | `stgcn` | 3,07 | 0,8621 | 0,8549 | 0,8620 |
| Skeleton | `sta_gcn` | 0,69 | 0,8324 | 0,8264 | 0,8327 |
| Skeleton | `ms_g3d` | 0,73 | 0,8029 | 0,7975 | 0,8028 |
| RGB | `mvit_v2_s` | 34,26 | **0,8715** | **0,8681** | 0,8681 |
| RGB | `r3d_18` | 33,18 | 0,8624 | 0,8639 | 0,8587 |
| RGB | `mc3_18` | 11,51 | 0,8637 | 0,8626 | 0,8593 |
| RGB | `videomae` | 86,25 | 0,8366 | 0,8318 | 0,8320 |
| RGB | `s3d` | 7,95 | 0,8304 | 0,8292 | 0,8256 |
| RGB | `videomae2` | 21,89 | 0,8282 | 0,8250 | 0,8263 |

Kiến trúc có Macro-F1 cao nhất trên từng protocol:

| Nhánh | Protocol | Kiến trúc | Accuracy | Macro-F1 | Tham số (M) |
|---|---|---|---:|---:|---:|
| IMU | `x_scene` | `tcn` | 0,7216 | 0,7251 | 0,21 |
| IMU | `x_view` | `tcn` | 0,8197 | 0,8211 | 0,21 |
| IMU | `x_sub_scene` | `inceptiontime` | 0,8033 | 0,8034 | 0,48 |
| IMU | `x_sub_view` | `tcn` | 0,6897 | 0,6798 | 0,21 |
| IMU | `x_session_scene` | `tcn` | 0,8219 | 0,8240 | 0,21 |
| IMU | `x_session_view` | `tcn` | 0,7920 | 0,7873 | 0,21 |
| Skeleton | `x_scene` | `stgcn` | 0,8619 | 0,8599 | 3,07 |
| Skeleton | `x_view` | `ctrgcn_head` | 0,7849 | 0,7495 | 2,49 |
| Skeleton | `x_sub_scene` | `ctrgcn_head` | 0,9116 | 0,9089 | 2,49 |
| Skeleton | `x_sub_view` | `ctrgcn_head` | 0,8974 | 0,8957 | 2,49 |
| Skeleton | `x_session_scene` | `ctrgcn_head` | 0,9244 | 0,9252 | 2,49 |
| Skeleton | `x_session_view` | `ctrgcn_head` | 0,9169 | 0,9176 | 2,49 |
| RGB | `x_scene` | `mvit_v2_s` | 0,8228 | 0,8192 | 34,26 |
| RGB | `x_view` | `r3d_18` | 0,7914 | 0,7850 | 33,18 |
| RGB | `x_sub_scene` | `mvit_v2_s` | 0,9031 | 0,9018 | 34,26 |
| RGB | `x_sub_view` | `mc3_18` | 0,8925 | 0,9006 | 11,51 |
| RGB | `x_session_scene` | `mc3_18` | 0,9179 | 0,9224 | 11,51 |
| RGB | `x_session_view` | `mvit_v2_s` | 0,9060 | 0,9101 | 34,26 |

Theo Macro-F1 trung bình, `tcn` dẫn đầu IMU (0,7729), `ctrgcn_head` dẫn
đầu Skeleton (0,8739), và `mvit_v2_s` dẫn đầu RGB (0,8681). `mc3_18` có
Macro-F1 RGB trung bình 0,8626 và Accuracy 0,8637. Theo cả hai metric,
`mc3_18` đứng nhất ở 2/6 protocol, nhì ở 2/6 và ba ở 2/6; do đó nó nằm
trong top 3 ở cả sáu protocol. So với `mvit_v2_s` dẫn đầu trung bình,
Macro-F1 thấp hơn 0,0055 và Accuracy thấp hơn 0,0078, trong khi số tham số
chỉ là 11,51M so với 34,26M (ít hơn 66,4%, tương đương gần 3 lần gọn hơn).
Vì vậy, chốt `mc3_18` làm backbone RGB cho cấu hình fusion: đây là lựa chọn
cân bằng giữa độ ổn định qua protocol, chất lượng validation và độ gọn,
không phải model có Macro-F1 trung bình cao nhất.

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
5. Đánh giá trên validation; quy trình sáu protocol hiện không tạo internal
   test split.

Các con số OOF/validation cuối cùng chỉ có thể xác định sau khi upload dữ liệu
hoàn tất. Số clip có thể khác nhau giữa các modality do độ phủ dữ liệu khác
nhau và fusion căn chỉnh dự đoán theo giao `clip_id`.
OOF bảo vệ bước fit fusion khỏi dự đoán train in-sample; nó không tự chọn
kiến trúc base model thay cho validation benchmark.

## 8. Kết quả fusion lịch sử (legacy)

Các số liệu trong mục này được tạo bởi quy trình split cũ và chỉ lưu làm
tham chiếu lịch sử. Chúng không phải kết quả của sáu protocol mới, không
được dùng để so sánh giữa các protocol, và code hiện tại không còn tạo
internal test split.

### 8.1 Giao thức đánh giá cũ

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

### 8.2 Kết quả từng modality trên giao test cũ

| Thí nghiệm | Modality | Phương pháp | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---:|---:|---:|
| E1 | Skeleton | Single | 0,9271 | 0,9247 | 0,9283 |
| E2 | IMU | Single | 0,7801 | 0,7737 | 0,7803 |
| E3 | RGB | Single | 0,8660 | 0,8789 | 0,8679 |

Ở giao test hiện tại, IMU đơn modality thấp hơn skeleton và RGB. Đây là điểm
cần được thảo luận như một hạn chế/quan sát thực nghiệm; không nên thay bằng
kết quả từ một benchmark khác dùng split hoặc số mẫu khác.

### 8.3 Kết quả tổ hợp hai modality cũ

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

### 8.4 Kết quả tổ hợp ba modality cũ

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

### 8.5 Diễn giải lịch sử

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

## 9. Hạn chế của kết quả legacy

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
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không có')"
```

Lệnh trên cài PyTorch kèm runtime CUDA 12.8; không cần cài CUDA Toolkit/`nvcc`
để chạy model. Nếu dùng GPU NVIDIA khác, chọn wheel phù hợp tại trang cài đặt
chính thức của PyTorch. Các thư viện EDA, xử lý ảnh và collector được khai báo
trong `requirements.txt`. `decord` chỉ cài trên Python dưới 3.13; Python 3.13
dùng OpenCV để đọc video.

### 10.2 Tạo sáu protocol

Chuẩn bị manifest nguồn cho hai split chính thức, gắn video RGB, sau đó sinh
train/val manifest riêng cho từng modality:

```bash
python tools/prepare_mmact.py --split cross_view
python tools/prepare_mmact.py --split cross_scene
python tools/prepare_rgb_mmact.py
python scripts/04_generate_protocol_manifests.py --seed 42 --upload-complete
```

Hai split chính thức được giữ nguyên: `x_view` lấy train/val từ cross-view,
`x_scene` từ cross-scene. Bốn split còn lại dùng 80/20 `StratifiedGroupKFold`
với group lần lượt là `(subject, scene)`, `(subject, cam)`,
`(subject, scene, session)` và `(subject, cam, session)`.

Output nằm ở `data/processed/MMAct/manifests/protocols/<protocol>/`:
`rgb/train.csv`, `skeleton/train.csv`, `imu/train.csv` và các file `val.csv`
tương ứng. `alignment/` chỉ là index để ghép dự đoán fusion; training luôn
đọc manifest riêng modality. Không có internal test split. Generator kiểm tra
đường dẫn file, group overlap và IMU event overlap; IMU trùng physical event
được dedupe và giữ ở val nếu cùng event xuất hiện hai phía.

Dataset còn đang upload: có thể chạy generator không kèm
`--upload-complete` để tạo snapshot tạm; audit sẽ đánh dấu `INCOMPLETE`, và
benchmark/OOF sẽ từ chối các manifest đó nếu không được override rõ ràng.
Sau khi xác nhận upload hoàn tất, chạy lại với `--upload-complete` để đánh
dấu manifest chính thức. Các đường dẫn modality thiếu vẫn được lọc và ghi
coverage vào audit.

### 10.3 Chạy EDA

```bash
python scripts/05_run_eda.py \
  --protocol x_sub_scene \
  --subset all \
  --manifest-dir data/processed/MMAct/manifests/protocols \
  --output-root outputs/eda
```

`x_sub_scene` gộp dữ liệu từ hai split nguồn; `--subset all` gộp train và val
để thống kê toàn bộ clip của protocol được chọn. Các phép kiểm tra file-level
trong module vẫn dùng số mẫu kiểm tra đã cấu hình; EDA clip hiện tại chưa phân
tích session-level untrimmed.

### 10.4 Huấn luyện modality

Ví dụ train kiến trúc đang dùng cho fusion:

```bash
python scripts/13_benchmark_models.py \
  --modality skeleton --mode train --protocol x_view \
  --architectures ctrgcn_head \
  --manifest-dir data/processed/MMAct/manifests/protocols/x_view/skeleton

python scripts/13_benchmark_models.py \
  --modality imu --mode train --protocol x_view \
  --architectures tcn \
  --manifest-dir data/processed/MMAct/manifests/protocols/x_view/imu

python scripts/13_benchmark_models.py \
  --modality rgb --mode train --protocol x_view \
  --architectures mc3_18 \
  --epochs 10 \
  --manifest-dir data/processed/MMAct/manifests/protocols/x_view/rgb
```

Thay `x_view` bằng một trong sáu protocol để chạy trên split tương ứng.
Benchmark RGB mặc định chạy 10 epoch mỗi kiến trúc; có thể đổi bằng `--epochs`.

### 10.5 Sinh OOF và chạy fusion trên cả sáu protocol

Runner lần lượt lấy checkpoint của các kiến trúc đã chốt trong cấu hình,
sinh OOF riêng theo từng protocol rồi chạy voting và attention trên validation:

```bash
python scripts/17_run_all_protocol_fusion.py \
  --protocol all \
  --stage both \
  --num-workers 8
```

Runner hỗ trợ truyền epoch override cho mô hình RGB trong từng OOF fold, ví
dụ `--rgb-epochs 25`. Nếu thay đổi số epoch của base model, cần benchmark lại
RGB trên cả sáu protocol với cùng số epoch trước; OOF và checkpoint dùng đánh
giá validation nên phản ánh cùng thiết lập.

Mặc định runner yêu cầu manifest audit ở trạng thái `PASS`. Có thể kiểm tra
checkpoint và lệnh mà runner sẽ chạy trước khi bắt đầu huấn luyện:

```bash
python scripts/17_run_all_protocol_fusion.py --protocol all --stage both --dry-run
```

Nếu OOF đã có đủ ba artifact cho từng modality thì bước sinh OOF được bỏ qua;
dùng `--force-oof` để tạo lại. Có thể chạy riêng `--stage oof` hoặc
`--stage fusion`; stage fusion yêu cầu OOF đã đầy đủ. Nếu audit chưa `PASS`,
chỉ thêm `--allow-incomplete-audit` khi đã chủ động chấp nhận kết quả trên
snapshot hiện tại chưa được xác nhận upload đầy đủ.

Mỗi protocol lưu fold logs và summary dưới `outputs/oof/<protocol>/`, đồng
thời ghi các dự đoán OOF vào đúng ba checkpoint đã chọn trong
`outputs/runs/{skeleton,imu,rgb}/`. Fusion lưu từng run riêng dưới
`outputs/runs/fusion/`; tất cả phương pháp được báo cáo riêng trên validation,
không có internal test split và không tự chọn “best” bằng điểm validation.

## 11. Cấu trúc mã nguồn và artifact

```text
configs/                 Cấu hình lớp, dữ liệu, train, fusion
data/raw/README.md       Cấu trúc và kiểm kê snapshot dữ liệu raw
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

Các bảng kết quả phía trên là lịch sử của split cũ. Sau khi hoàn tất upload,
cần tạo lại sáu protocol, chạy benchmark và báo cáo kết quả mới cùng audit
coverage/leakage của từng modality. Không dùng số liệu legacy để kết luận
protocol nào tốt hơn.

## 13. Theo dõi camera qua giao diện web

Chạy web server để xem luồng RTSP trên trình duyệt:

```bash
export RTSP_URL='rtsp://<user>:<password>@192.168.167.163:554/stream1'
python vison.py
```

Mở `http://127.0.0.1:8081`. Có thể truyền URL bằng `--source` thay cho biến
môi trường. Mặc định server chỉ nghe trên máy local; khi dùng VS Code Remote,
forward cổng `8081` để mở giao diện trên máy cá nhân. Có thể đặt `--host
0.0.0.0` để truy cập qua mạng, nhưng giao diện chưa có xác thực nên chỉ dùng
trên mạng đáng tin cậy. Không ghi URL RTSP chứa mật khẩu vào mã nguồn hoặc
chia sẻ URL đó.

Camera `192.168.167.163:554` cần xác thực RTSP; nếu không có thông tin đăng
nhập, camera trả về `401 Unauthorized`. Hãy thay `<user>` và `<password>`
bằng tài khoản camera của bạn. Nếu mật khẩu chứa ký tự đặc biệt như `@`, `:`,
`/` hoặc `#`, cần URL-encode phần tên đăng nhập/mật khẩu. Mặc định chương
trình dùng RTSP qua TCP và giới hạn thời gian chờ mở/đọc luồng để có thể tự
kết nối lại khi camera gián đoạn.
