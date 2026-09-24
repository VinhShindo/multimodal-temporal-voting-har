Đúng hướng. Nếu lấy **CoboGesture làm khung phương pháp chính**, nhưng mở rộng từ RGB/Skeleton thành **RGB + Skeleton + IMU**, thì nên xây dựng hệ thống theo kiểu:

> **Huấn luyện riêng từng modality → chuẩn hóa đầu ra về cùng một không gian lớp + timestamp → kết hợp bằng voting → temporal post-processing → đánh giá isolated và continuous.**

Điểm quan trọng là **không cố ép 3 loại dữ liệu dùng cùng một backbone**. Cái cần thống nhất là **class space, thời gian, format đầu ra và giao thức đánh giá**.

Trong CoboGesture, RGB-CoGes và SkeRGB-CoGes dùng chung mô hình nhận dạng isolated; khác nhau chủ yếu ở cách lấy đoạn video khi inference: RGB-CoGes dùng sliding window, còn SkeRGB-CoGes dùng skeleton để xác định boundary rồi mới gọi mô hình RGB. Bài báo cũng dùng smoothing → linking → removing cho RGB-CoGes và majority voting trên các cửa sổ trong một đoạn đã xác định cho SkeRGB-CoGes.  

Với đề tài của bạn, tôi đề xuất **mở rộng thành một framework 3-modal** như sau.

---

# 1. Kiến trúc tổng thể đề xuất

```text
                    DATA COLLECTION
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
    RGB Video          Skeleton             IMU
    Camera             Pose                 Smartwatch
        │                  │                  │
        └────────────┬─────┴─────┬───────────┘
                     │
                     ▼
              DATA SYNCHRONIZATION
                     │
                     ▼
              DATA PREPROCESSING
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   RGB Dataset   Skeleton Dataset  IMU Dataset
       │             │             │
       ▼             ▼             ▼
    RGB Model    Skeleton Model   IMU Model
       │             │             │
       ▼             ▼             ▼
     P_RGB        P_Skeleton      P_IMU
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              OUTPUT ALIGNMENT
                     │
                     ▼
              MODALITY VOTING
          ┌──────────┼──────────┐
          │          │          │
       RGB only   RGB+SK    RGB+SK+IMU
          │          │          │
          └──────────┼──────────┘
                     ▼
              TEMPORAL VOTING
                     │
                     ▼
              POST-PROCESSING
                     │
                     ▼
              FINAL ACTIVITIES
```

Tôi sẽ chia việc triển khai thành **8 giai đoạn**.

---

# 2. Giai đoạn 1 — Xác định bộ action/classes

Trước khi thu thập dữ liệu, phải cố định **một bộ class chung** cho cả 3 model.

Ví dụ nếu bạn đang làm human activity recognition:

```text
0   standing
1   sitting
2   lying
3   walking
4   running
5   standing_up
6   sitting_down
7   drinking
8   using_phone
9   picking_object
10  carrying_object
11  waving_hand
12  falling
```

Có thể thêm:

```text
13  background
```

hoặc:

```text
13  no_action
```

### Rất quan trọng

Cả 3 model phải dùng **cùng mapping**:

```text
class_id = 0 → standing
class_id = 1 → sitting
...
class_id = 12 → falling
```

Không được để:

```text
RGB:
0 = standing
1 = sitting

IMU:
0 = walking
1 = standing
```

Nếu khác mapping thì voting sẽ sai hoàn toàn.

---

# 3. Giai đoạn 2 — Thu thập dữ liệu

Đây là phần tôi khuyên bạn làm giống tư duy của CoboGesture nhất.

CoboGesture không chỉ thu isolated clip mà thu **untrimmed video liên tục**, sau đó annotation start/end của từng gesture. Dataset của họ có 150 video từ 50 subject, mỗi video khoảng 100–140 giây và chứa nhiều gesture; giữa các gesture có khoảng nghỉ. 

Với đề tài của bạn cũng nên làm tương tự.

## 3.1. Không nên chỉ quay từng action riêng biệt

Không nên chỉ:

```text
standing.mp4
walking.mp4
sitting.mp4
running.mp4
```

vì như vậy bạn chỉ đánh giá isolated recognition.

Nên có:

```text
continuous_session_001.mp4

standing
↓
walking
↓
sitting
↓
using_phone
↓
walking
↓
falling
↓
standing
```

Như vậy bạn mới đánh giá được:

* action classification
* temporal localization
* start/end
* continuous recognition
* voting
* post-processing.

---

# 4. Một session nên thu thập những gì?

Mỗi session nên đồng thời ghi:

```text
                    SESSION
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
     CAMERA         SKELETON           IMU
       │               │                │
    RGB video      3D keypoints      Acc/Gyro
```

### Camera

Ví dụ:

```text
resolution: 1280×720
FPS: 30
format: MP4
```

CoboGesture dùng camera 30 FPS và annotation trực tiếp trong lúc quay; camera của họ đặt cách subject khoảng 2.5 m. Đây có thể dùng làm tham khảo về cách tổ chức data collection, không nhất thiết phải sao chép nguyên thông số. 

### Skeleton

Không nhất thiết phải lưu video skeleton.

Nên lưu:

```text
frame_id
timestamp
joint_id
x
y
z
confidence
```

Ví dụ:

```text
frame_00001:
joint_0 = [x,y,z,confidence]
joint_1 = [x,y,z,confidence]
...
joint_J = [x,y,z,confidence]
```

### Smartwatch

Ghi:

```text
timestamp
accelerometer_x
accelerometer_y
accelerometer_z
gyro_x
gyro_y
gyro_z
```

Nếu smartwatch có orientation:

```text
roll
pitch
yaw
```

thì cũng có thể lưu.

---

# 5. Annotation — đây là phần cực kỳ quan trọng

Bạn nên có **một annotation gốc cho toàn bộ session**.

Ví dụ:

```csv
start_time,end_time,label
0.0,3.2,standing
3.3,7.8,walking
8.1,11.5,sitting
12.0,16.2,using_phone
17.0,19.0,falling
20.0,24.0,standing
```

Từ file annotation này bạn tạo ra:

```text
RGB labels
Skeleton labels
IMU labels
```

chứ **không annotation độc lập 3 lần**.

---

# 6. Đồng bộ thời gian — bước bắt buộc

Đây là điểm cực kỳ quan trọng khi thêm IMU.

Giả sử camera:

```text
30 FPS
```

và smartwatch:

```text
100 Hz
```

thì không thể có:

```text
frame 100 ↔ IMU sample 100
```

vì hai sensor có sampling rate khác nhau.

Phải dùng:

```text
timestamp
```

làm chuẩn.

Ví dụ:

```text
Camera

00:05.000
00:05.033
00:05.067
00:05.100
...
```

IMU:

```text
00:05.000
00:05.010
00:05.020
00:05.030
...
```

Sau đó:

```text
        MASTER TIMELINE
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
     RGB   Skeleton    IMU
```

---

# 7. Chuẩn hóa dữ liệu từng modality

Không nên chuẩn hóa RGB, Skeleton, IMU theo cùng một cách.

---

## 7.1. RGB

Pipeline:

```text
Video
 ↓
Decode frames
 ↓
Resize
 ↓
Center crop / spatial crop
 ↓
Normalize pixel
 ↓
Temporal sampling
 ↓
T frames
 ↓
RGB Model
```

Theo CoboGesture, input video có \(\tau\) frame rồi lấy \(T\) frame bằng temporal sampling với stride = 4; họ dùng VideoMAE/VideoMAEv2 với ViT-L/16 và thay decoder bằng lớp phục vụ classification.  

Bạn có thể bắt đầu với:

```text
window τ = 64 frames
sampling stride = 4
T = 16 frames
```

Đây cũng là cấu hình được CoboGesture dùng trong thí nghiệm của họ. 

---

# 8. Skeleton preprocessing

Skeleton có vấn đề lớn là:

> mỗi người có chiều cao, vị trí đứng và khoảng cách camera khác nhau.

Vì vậy không nên đưa trực tiếp:

```text
x,y,z
```

vào model.

Có thể chuẩn hóa theo:

### Bước 1 — chọn root

Ví dụ:

```text
hip / torso
```

Sau đó:

$$
x' = x-x_{root}
$$

$$
y' = y-y_{root}
$$

$$
z' = z-z_{root}
$$

### Bước 2 — scale

Có thể chia theo khoảng cách vai hoặc torso:

$$
p' = \frac{p-p_{root}}{scale}
$$

Kết quả:

```text
Skeleton
    ↓
Translation normalization
    ↓
Scale normalization
    ↓
Temporal interpolation
    ↓
Fixed-length sequence
    ↓
Skeleton Model
```

Nếu dùng skeleton từ camera, nên lưu cả:

```text
x
y
z
confidence
```

CoboGesture sử dụng MediaPipe và tạo skeleton mỗi frame dạng \(48\times3\), gồm 42 điểm bàn tay và 6 điểm phần thân trên. 

---

# 9. IMU preprocessing

IMU cần xử lý khác hẳn.

Raw:

```text
timestamp
ax ay az
gx gy gz
```

Có thể xử lý:

```text
Raw IMU
   ↓
Remove invalid samples
   ↓
Resampling
   ↓
Synchronization
   ↓
Normalization
   ↓
Sliding window
   ↓
IMU Model
```

Ví dụ:

```text
Accelerometer:
ax, ay, az

Gyroscope:
gx, gy, gz
```

Normalize:

$$
x' = \frac{x-\mu}{\sigma}
$$

với \(\mu,\sigma\) **chỉ tính từ training set**.

Không được tính mean/std riêng cho test.

---

# 10. Cách chia dataset

Đây là phần cực kỳ quan trọng để tránh data leakage.

Không nên:

```text
Subject 01
 ├── 80% train
 └── 20% test
```

vì model có thể học đặc điểm riêng của người đó.

Nên:

```text
Subjects
│
├── Train subjects
├── Validation subjects
└── Test subjects
```

Ví dụ:

```text
Subject 01–24 → train
Subject 25–32 → validation
Subject 33–40 → test
```

Tỷ lệ chỉ là ví dụ.

Quan trọng là:

> **Một subject chỉ xuất hiện ở một split.**

CoboGesture cũng đánh giá theo subject ID, dùng subject ID lẻ cho training và ID chẵn cho testing. 

---

# 11. Tạo dữ liệu training cho 3 model

Sau khi có continuous session:

```text
continuous video
       +
continuous skeleton
       +
continuous IMU
       +
annotation
```

ta tạo isolated samples.

Ví dụ:

```text
3.3s → 7.8s = walking
```

tạo:

```text
RGB sample:
3.3 → 7.8

Skeleton sample:
3.3 → 7.8

IMU sample:
3.3 → 7.8
```

Tất cả có:

```text
label = walking
```

---

# 12. Training Model 1 — RGB

Tôi đề xuất:

```text
RGB
 ↓
VideoMAE / VideoMAEv2
 ↓
Classifier
 ↓
Softmax
 ↓
P_RGB
```

Ví dụ:

```text
Input:
[B, C, T, H, W]

Output:
[B, N]
```

Trong đó:

```text
N = số action classes
```

Ví dụ 14 class:

```text
P_RGB =
[
  0.02,
  0.01,
  0.00,
  0.82,
  ...
]
```

và:

```text
argmax(P_RGB)
= walking
```

CoboGesture dùng VideoMAE/VideoMAEv2 như isolated classifier và thay decoder reconstruction bằng classification layer + softmax. 

---

# 13. Training Model 2 — Skeleton

Đây là model **độc lập**.

Tôi đề xuất:

```text
Skeleton
   ↓
CTR-GCN
   ↓
Global pooling
   ↓
Linear
   ↓
Softmax
   ↓
P_Skeleton
```

hoặc:

```text
Skeleton
   ↓
ST-GCN
   ↓
Classifier
```

CoboGesture cũng benchmark các mô hình skeleton như ST-GCN và CTR-GCN. 

---

# 14. Training Model 3 — IMU

IMU là time-series nên không dùng VideoMAE.

Có thể bắt đầu:

```text
IMU
 ↓
1D CNN
 ↓
BiLSTM
 ↓
Global Pooling
 ↓
Linear
 ↓
Softmax
 ↓
P_IMU
```

hoặc nếu muốn benchmark mạnh hơn:

```text
IMU
 ↓
TCN / InceptionTime
 ↓
Classifier
```

Tôi khuyên ban đầu dùng:

**1D CNN + BiLSTM**

vì dễ giải thích trong đồ án.

---

# 15. Điểm quan trọng nhất: chuẩn hóa OUTPUT

Đây mới là nơi 3 model gặp nhau.

Dù architecture khác nhau:

```text
VideoMAE
CTR-GCN
CNN+LSTM
```

thì output phải giống format.

Ví dụ:

```text
RGB Model
    ↓
P_RGB ∈ R^14

Skeleton Model
    ↓
P_SK ∈ R^14

IMU Model
    ↓
P_IMU ∈ R^14
```

và:

$$
\sum_i P_i = 1
$$

---

# 16. Chuẩn hóa class order

Tạo một file duy nhất:

```yaml
classes:
  0: standing
  1: sitting
  2: lying
  3: walking
  4: running
  5: standing_up
  6: sitting_down
  7: drinking
  8: using_phone
  9: picking_object
  10: carrying_object
  11: waving_hand
  12: falling
  13: no_action
```

Tất cả model đọc file này.

```text
RGB      ──┐
Skeleton ──┼──> SAME CLASS MAP
IMU      ──┘
```

---

# 17. Chuẩn hóa timestamp output

Mỗi prediction phải có:

```json
{
  "start_time": 10.0,
  "end_time": 12.0,
  "class_id": 3,
  "class_name": "walking",
  "confidence": 0.86
}
```

Nếu dùng probability:

```json
{
  "start_time": 10.0,
  "end_time": 12.0,
  "probabilities": [
    0.01,
    0.02,
    0.00,
    0.86,
    0.04
  ]
}
```

Như vậy mới có thể fusion.

---

# 18. Inference — cách 1: RGB-only

Đây là baseline tương ứng với **RGB-CoGes**.

```text
Continuous RGB
       ↓
Sliding Window
       ↓
RGB Model
       ↓
P_RGB
       ↓
Temporal sequence
       ↓
Smoothing
       ↓
Linking
       ↓
Removing
       ↓
Final Activity
```

CoboGesture dùng sliding window với overlap ít nhất 50%; sau đó smoothing, linking và removing để tạo kết quả continuous. 

Ví dụ:

```text
Window 1 → walking
Window 2 → walking
Window 3 → walking
Window 4 → sitting
Window 5 → sitting
```

---

# 19. Inference — Skeleton-only

Đây là baseline thứ hai.

```text
Skeleton
   ↓
Sliding Window
   ↓
CTR-GCN
   ↓
P_SK
   ↓
Temporal post-processing
   ↓
Final Activity
```

Bạn có thể đánh giá:

```text
ST-GCN
CTR-GCN
```

để làm baseline skeleton.

CoboGesture cũng dùng DD-Net, TDDNet, ST-GCN và CTR-GCN làm các phương pháp so sánh cho continuous recognition. 

---

# 20. Inference — IMU-only

```text
Continuous IMU
       ↓
Sliding Window
       ↓
IMU Model
       ↓
P_IMU
       ↓
Temporal Voting
       ↓
Post-processing
       ↓
Final Activity
```

Đây là baseline mới của đề tài bạn, **không phải thành phần của CoboGesture gốc**.

---

# 21. RGB + Skeleton — mô phỏng tư tưởng SkeRGB-CoGes

Có hai cách.

### Cách A — Skeleton làm boundary detector

Đây là cách sát với CoboGesture nhất.

```text
Continuous Skeleton
        ↓
Boundary Detection
        ↓
start_frame
        ↓
RGB segment
        ↓
VideoMAE
        ↓
P_RGB
        ↓
Voting
        ↓
Final Action
```

CoboGesture dùng skeleton để xác định điểm bắt đầu/kết thúc rồi mới gọi VideoMAE trên đoạn RGB tương ứng. 

---

# 22. RGB + Skeleton — nếu muốn fusion thực sự

Bạn cũng có thể làm:

```text
RGB Window
     ↓
VideoMAE
     ↓
P_RGB

Skeleton Window
     ↓
CTR-GCN
     ↓
P_SK

P_RGB + P_SK
     ↓
Fusion
     ↓
Final
```

Đây là **multimodal voting**, khác với SkeRGB-CoGes gốc.

---

# 23. RGB + IMU

Tương tự:

```text
RGB
 ↓
VideoMAE
 ↓
P_RGB

IMU
 ↓
IMU Model
 ↓
P_IMU

     ↓
Voting
     ↓
Final
```

---

# 24. Skeleton + IMU

```text
Skeleton
 ↓
CTR-GCN
 ↓
P_SK

IMU
 ↓
IMU Model
 ↓
P_IMU

     ↓
Voting
     ↓
Final
```

---

# 25. RGB + Skeleton + IMU

Đây mới là hệ thống chính của bạn:

```text
                   RGB
                    │
                VideoMAE
                    │
                  P_RGB
                    │
                    │
Skeleton ──→ CTR-GCN ──→ P_SK
                    │
                    │
IMU ─────→ IMU Model ──→ P_IMU
                    │
             ┌──────┴──────┐
             │              │
             ▼              ▼
        Probability      Hard Label
          Voting           Voting
             │              │
             └──────┬───────┘
                    ▼
              Temporal Voting
                    ↓
              Post-processing
                    ↓
             Final Activities
```

---

# 26. Voting 1 — Hard Majority Voting

Mỗi model chọn một class.

Ví dụ:

```text
RGB       → walking
Skeleton  → walking
IMU       → running
```

Voting:

```text
walking = 2
running = 1
```

→ final:

```text
walking
```

Công thức:

$$
\hat y =
\arg\max_c
\sum_m
\mathbf{1}(\hat y_m=c)
$$

Trong đó:

```text
m = RGB, Skeleton, IMU
```

---

# 27. Nhưng hard voting có nhược điểm

Ví dụ:

```text
RGB:
walking = 0.95

Skeleton:
running = 0.51

IMU:
running = 0.52
```

Hard voting:

```text
running = 2
walking = 1
```

nhưng rõ ràng RGB có confidence rất cao.

Vì vậy tôi khuyên bạn benchmark **cả hard voting và probability voting**.

---

# 28. Voting 2 — Probability Voting

Thay vì chỉ lấy class:

```text
walking
```

ta giữ toàn bộ vector:

```text
P_RGB
P_SK
P_IMU
```

Sau đó:

$$
P_{fusion}
=
\frac{
P_{RGB}+P_{SK}+P_{IMU}
}{3}
$$

Ví dụ:

```text
             walking running sitting
RGB           0.80    0.10    0.10
Skeleton      0.60    0.30    0.10
IMU           0.30    0.60    0.10
------------------------------------
Average       0.57    0.33    0.10
```

Final:

```text
walking
```

---

# 29. Voting 3 — Weighted Probability Voting

Đây là phương pháp tôi khuyên bạn đưa vào phần **ablation/extension**.

$$
P_{fusion}
=
w_RP_{RGB}
+
w_SP_{Skeleton}
+
w_IP_{IMU}
$$

với:

$$
w_R+w_S+w_I=1
$$

Ví dụ:

```text
RGB       = 0.40
Skeleton  = 0.30
IMU       = 0.30
```

thì:

$$
P_{fusion}
=
0.4P_{RGB}
+
0.3P_{SK}
+
0.3P_{IMU}
$$

**Không nên tự khẳng định 0.4/0.3/0.3 là tối ưu.**

Bạn phải tìm weight trên validation set.

Ví dụ thử:

```text
1/3, 1/3, 1/3

0.4, 0.3, 0.3

0.5, 0.25, 0.25

0.5, 0.3, 0.2

0.6, 0.2, 0.2
```

và chọn theo validation metric.

---

# 30. Một vấn đề cần giải quyết: 3 model không nhất thiết dự đoán cùng lúc

Ví dụ:

```text
RGB:
10.0 → 12.0

Skeleton:
10.0 → 12.0

IMU:
10.0 → 12.0
```

thì rất dễ.

Nhưng thực tế:

```text
RGB:       30 FPS
Skeleton:  30 FPS
IMU:       100 Hz
```

Do đó phải có:

```text
Timestamp Alignment
```

trước voting.

---

# 31. Chuẩn hóa về common temporal window

Ví dụ chọn:

```text
window = 2 seconds
step = 0.5 seconds
```

thì:

```text
Window 1:
0.0 → 2.0

Window 2:
0.5 → 2.5

Window 3:
1.0 → 3.0
```

Cả ba modality đều phải sinh prediction cho:

```text
0.0 → 2.0
0.5 → 2.5
1.0 → 3.0
```

Sau đó:

```text
RGB Window 1
Skeleton Window 1
IMU Window 1
       ↓
    Fusion 1
```

---

# 32. Cấu trúc output nên như thế này

```json
{
  "window_id": 15,
  "start": 7.5,
  "end": 9.5,

  "rgb": {
    "class_id": 3,
    "confidence": 0.86,
    "probabilities": [...]
  },

  "skeleton": {
    "class_id": 3,
    "confidence": 0.78,
    "probabilities": [...]
  },

  "imu": {
    "class_id": 4,
    "confidence": 0.62,
    "probabilities": [...]
  },

  "fusion": {
    "class_id": 3,
    "confidence": 0.75
  }
}
```

Đây sẽ là interface rất tốt giữa các model.

---

# 33. Temporal Voting

Sau modality voting bạn vẫn chưa nên xuất ngay:

```text
walking
walking
running
walking
walking
```

vì sẽ có noise.

Ví dụ:

```text
Window:

W1 walking
W2 walking
W3 walking
W4 running
W5 walking
W6 walking
```

Temporal voting:

```text
[walking, walking, running, walking, walking]
```

→

```text
walking
```

---

# 34. Temporal majority voting

Có thể dùng một voting buffer:

```text
N = 5 windows
```

Ví dụ:

```text
Buffer:
walking
walking
running
walking
walking
```

→

```text
walking
```

Đây cũng phù hợp với cách majority voting được mô tả trong phần related work của CoboGesture: giữ các prediction trong một time window và chọn class có số vote cao nhất. 

---

# 35. Post-processing theo CoboGesture

Đối với pipeline RGB-based, tôi khuyên giữ đúng 3 bước:

```text
Prediction
    ↓
Smoothing
    ↓
Linking
    ↓
Removing
    ↓
Final segments
```

CoboGesture định nghĩa smoothing bằng cách lấy trung bình xác suất của frame hiện tại với frame trước và sau; linking nối các segment cùng nhãn nếu khoảng cách giữa chúng không vượt quá \(t_{avg}\); removing loại các đoạn ngắn/nhiễu. 

---

# 36. Smoothing

Ví dụ:

```text
t1 = walking 0.80
t2 = walking 0.82
t3 = running 0.55
t4 = walking 0.79
```

Thay vì dùng trực tiếp:

```text
walking
walking
running
walking
```

có thể smooth probability theo temporal neighborhood.

---

# 37. Linking

Ví dụ:

```text
walking
walking

[gap 0.4s]

walking
walking
```

Nếu:

```text
gap < t_avg
```

thì:

```text
walking ───────────── walking
```

→ gộp thành một action.

CoboGesture chọn \(t_{avg}\) dựa trên validation set và định nghĩa nó liên quan đến thời lượng chuyển động trung bình của các class. 

---

# 38. Removing

Ví dụ:

```text
walking: 4s
running: 0.2s
walking: 3s
```

Nếu:

```text
running duration < threshold
```

thì loại:

```text
walking
walking
walking
```

Cách này giúp giảm prediction noise.

---

# 39. SkeRGB-CoGes-style pipeline cho hệ thống của bạn

Nếu muốn **bám sát CoboGesture nhất**, tôi sẽ triển khai pipeline chính như sau:

```text
             CONTINUOUS INPUT
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       RGB       Skeleton       IMU
        │           │           │
        │           ▼           │
        │      Boundary         │
        │      Detection        │
        │           │           │
        └───────────┼───────────┘
                    ▼
             START DETECTED
                    │
                    ▼
             ACTION SEGMENT
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        RGB       Skeleton    IMU
          │         │         │
       VideoMAE  CTR-GCN    IMU Model
          │         │         │
        P_RGB     P_SK      P_IMU
          │         │         │
          └─────────┼─────────┘
                    ▼
             MODALITY VOTING
                    │
                    ▼
              ACTION LABEL
                    │
                    ▼
            NEXT ACTION
```

Nhưng có một điểm:

**Skeleton boundary detection chỉ nên là một module phụ để tối ưu continuous inference.**

Bạn vẫn nên benchmark một pipeline thuần sliding-window để có baseline công bằng.

---

# 40. Do đó hệ thống của bạn nên có 3 tầng

## Tầng 1 — Isolated Recognition

```text
RGB clip
    ↓
RGB Model
    ↓
Accuracy

Skeleton clip
    ↓
Skeleton Model
    ↓
Accuracy

IMU clip
    ↓
IMU Model
    ↓
Accuracy
```

---

## Tầng 2 — Multimodal Fusion

```text
RGB
Skeleton
IMU
 ↓
Probability Alignment
 ↓
Voting
 ↓
Prediction
```

Benchmark:

```text
RGB
Skeleton
IMU

RGB + Skeleton
RGB + IMU
Skeleton + IMU

RGB + Skeleton + IMU
```

---

## Tầng 3 — Continuous Recognition

```text
Continuous stream
       ↓
Sliding window
       ↓
Individual models
       ↓
Multimodal voting
       ↓
Temporal voting
       ↓
Post-processing
       ↓
Continuous action segments
```

---

# 41. Benchmark đầy đủ tôi đề xuất

Đây là phần rất quan trọng để đồ án có giá trị nghiên cứu.

Bạn nên có bảng experiment:

| ID | Phương pháp          | RGB | Skeleton | IMU | Fusion |
| -- | -------------------- | --: | -------: | --: | ------ |
| E1 | RGB-only             |   ✓ |          |     |        |
| E2 | Skeleton-only        |     |        ✓ |     |        |
| E3 | IMU-only             |     |          |   ✓ |        |
| E4 | RGB + Skeleton       |   ✓ |        ✓ |     | Voting |
| E5 | RGB + IMU            |   ✓ |          |   ✓ | Voting |
| E6 | Skeleton + IMU       |     |        ✓ |   ✓ | Voting |
| E7 | RGB + Skeleton + IMU |   ✓ |        ✓ |   ✓ | Voting |

Sau đó thêm:

```text
E8 = RGB + Skeleton + IMU + temporal voting

E9 = RGB + Skeleton + IMU + temporal voting + post-processing

E10 = RGB + Skeleton + IMU + skeleton boundary detection
```

---

# 42. Benchmark riêng cho voting

Không nên chỉ có một kiểu voting.

Có thể:

| Experiment | Fusion                                                   |
| ---------- | -------------------------------------------------------- |
| V1         | Hard majority                                            |
| V2         | Average probability                                      |
| V3         | Weighted probability                                     |
| V4         | Weighted probability + temporal voting                   |
| V5         | Weighted probability + temporal voting + post-processing |

Như vậy bạn chứng minh được:

```text
Model
 ↓
Fusion
 ↓
Temporal processing
 ↓
Final
```

mỗi tầng đóng góp như thế nào.

---

# 43. Benchmark isolated

Dùng:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

Ví dụ:

```text
                 Accuracy
RGB                88.2%
Skeleton           84.7%
IMU                81.5%
RGB+Skeleton       91.0%
RGB+IMU            90.1%
Skeleton+IMU       87.8%
RGB+SK+IMU         93.2%
```

**Các con số trên chỉ là ví dụ minh họa, không phải kết quả thực nghiệm.**

---

# 44. Benchmark continuous

Ở đây nên dùng tư duy của CoboGesture:

### Frame-wise Accuracy

$$
Accuracy_{frame}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{1}(P_t=G_t)
$$

CoboGesture sử dụng frame-wise accuracy cho continuous recognition. 

---

# 45. Temporal overlap / IoU

Với ground truth:

```text
GT:
10.0 → 15.0
```

Prediction:

```text
Pred:
10.5 → 14.5
```

thì:

$$
IoU=
\frac{|GT\cap Pred|}
{|GT\cup Pred|}
$$

CoboGesture cũng dùng temporal overlap dựa trên tỷ lệ intersection/union và thêm điều kiện sai lệch start/end trong ngưỡng cho phép. 

---

# 46. Latency

Đối với real-time:

```text
Action end
      ↓
Model receives enough data
      ↓
Prediction
      ↓
Final output
```

đo:

$$
Latency=t_{output}-t_{action-end}
$$

CoboGesture khi triển khai trên Jetson AGX Orin báo cáo khoảng 94% frame-wise accuracy và latency trung bình khoảng 1.3 s; đồng thời họ ghi nhận tốc độ MediaPipe khoảng 4.5 FPS và VideoMAE khoảng 2.5 FPS trong thiết lập đó. Đây là **kết quả của CoboGesture**, không phải mục tiêu mặc định cho hệ thống của bạn. 

---

# 47. FPS / computational cost

Nên đo:

```text
RGB model FPS
Skeleton model FPS
IMU model FPS
Fusion FPS
End-to-end FPS
```

và:

```text
GPU memory
CPU usage
GPU usage
latency
```

Đặc biệt so sánh:

```text
RGB-only
```

với:

```text
Skeleton boundary + RGB
```

vì đây là một trong những ý tưởng quan trọng của CoboGesture.

---

# 48. Một benchmark rất hay cho đồ án

Bạn có thể chứng minh **tác dụng của boundary detection**:

### Experiment A

```text
RGB
 ↓
Sliding Window
 ↓
VideoMAE
```

### Experiment B

```text
Skeleton
 ↓
Boundary Detection
 ↓
RGB
 ↓
VideoMAE
```

So sánh:

```text
Accuracy
Overlap
FPS
GPU time
Latency
```

Nếu B giảm lượng RGB inference nhưng vẫn giữ kết quả tốt, đó là một kết quả rất đáng đưa vào đồ án.

---

# 49. Dataset structure tôi đề xuất

```text
dataset/
│
├── raw/
│   ├── videos/
│   │   ├── subject_001/
│   │   │   ├── session_01.mp4
│   │   │   └── session_02.mp4
│   │   └── subject_002/
│   │
│   ├── imu/
│   │   ├── subject_001/
│   │   │   ├── session_01.csv
│   │   │   └── session_02.csv
│   │   └── subject_002/
│   │
│   └── skeleton/
│       ├── subject_001/
│       └── subject_002/
│
├── annotations/
│   ├── session_01.csv
│   ├── session_02.csv
│   └── ...
│
├── processed/
│   ├── rgb/
│   ├── skeleton/
│   └── imu/
│
├── isolated/
│   ├── rgb/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── skeleton/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── imu/
│       ├── train/
│       ├── val/
│       └── test/
│
├── continuous/
│   ├── train/
│   ├── val/
│   └── test/
│
├── splits/
│   ├── train_subjects.txt
│   ├── val_subjects.txt
│   └── test_subjects.txt
│
└── classes.yaml
```

---

# 50. Model output directory

Nên lưu riêng prediction để sau này có thể thử voting mà **không phải chạy lại model**.

```text
outputs/
│
├── rgb/
│   ├── test_predictions.json
│   └── probabilities.npy
│
├── skeleton/
│   ├── test_predictions.json
│   └── probabilities.npy
│
├── imu/
│   ├── test_predictions.json
│   └── probabilities.npy
│
├── fusion/
│   ├── rgb_skeleton/
│   ├── rgb_imu/
│   ├── skeleton_imu/
│   └── rgb_skeleton_imu/
│
└── temporal/
    ├── smoothing/
    ├── linking/
    └── removing/
```

Đây là cách rất nên làm.

Ví dụ model đã chạy xong:

```text
P_RGB
P_SK
P_IMU
```

Bạn có thể chạy 10 kiểu voting khác nhau chỉ bằng Python mà **không cần inference lại**.

---

# 51. Pipeline training hoàn chỉnh

```text
                 RAW DATA
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       RGB       Skeleton       IMU
        │           │           │
        ▼           ▼           ▼
   Preprocess   Preprocess   Preprocess
        │           │           │
        └───────────┼───────────┘
                    ▼
             TIME ALIGNMENT
                    │
                    ▼
             SUBJECT SPLIT
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      RGB Data   SK Data   IMU Data
          │         │         │
          ▼         ▼         ▼
      VideoMAE   CTR-GCN   CNN-LSTM
          │         │         │
          ▼         ▼         ▼
      P_RGB      P_SK      P_IMU
          │         │         │
          └─────────┼─────────┘
                    ▼
              SAVE OUTPUT
```

---

# 52. Pipeline inference hoàn chỉnh

```text
             CONTINUOUS STREAM
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
      RGB        Skeleton          IMU
       │             │             │
       ▼             ▼             ▼
 Sliding Window  Boundary/Window  Sliding Window
       │             │             │
       ▼             ▼             ▼
   RGB Model      SK Model       IMU Model
       │             │             │
       ▼             ▼             ▼
     P_RGB         P_SK          P_IMU
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              TIME ALIGNMENT
                     │
                     ▼
              MODALITY FUSION
                     │
                     ▼
             TEMPORAL VOTING
                     │
                     ▼
                SMOOTHING
                     │
                     ▼
                 LINKING
                     │
                     ▼
                REMOVING
                     │
                     ▼
             FINAL SEGMENTS
```

---

# 53. Tôi khuyên bạn chia đề tài thành 3 baseline + 1 proposed system

Để đồ án dễ triển khai và kết quả dễ chứng minh:

### Baseline 1

**RGB-only**

```text
RGB → VideoMAE → Sliding Window → Post-processing
```

Đây là baseline gần **RGB-CoGes**.

### Baseline 2

**Skeleton-only**

```text
Skeleton → CTR-GCN → Sliding Window → Post-processing
```

### Baseline 3

**IMU-only**

```text
IMU → CNN-BiLSTM → Sliding Window → Post-processing
```

### Proposed

**RGB + Skeleton + IMU**

```text
RGB → VideoMAE ──────┐
                     │
Skeleton → CTR-GCN ──┼→ Probability Voting
                     │
IMU → CNN-BiLSTM ────┘
                     ↓
              Temporal Voting
                     ↓
              Post-processing
                     ↓
             Final Activity
```

Và nếu muốn bám CoboGesture sâu hơn:

```text
Skeleton
    ↓
Boundary Detection
    ↓
Activate RGB / IMU inference
    ↓
Multimodal Voting
```

---

# 54. Một điểm tôi muốn chỉnh so với mô tả ban đầu của bạn

Bạn viết:

> “Cách 2 ... Majority Voting ... không cần các bước hậu xử lý phức tạp.”

Điều này đúng với **SkeRGB-CoGes của CoboGesture** ở mức mô tả phương pháp: sau khi boundary detection xác định segment, họ dùng voting để chọn class cuối thay vì pipeline post-processing của RGB-CoGes. 

Nhưng với **đề tài mở rộng 3 modality của bạn**, tôi khuyên:

```text
Boundary Detection
        ↓
Multimodal Voting
        ↓
Temporal Voting
        ↓
Light Post-processing
```

thay vì bỏ hoàn toàn temporal processing.

Lý do là IMU có thể tạo prediction nhiễu và ba sensor có latency/sampling khác nhau. Đây là **đề xuất triển khai của tôi**, không phải nội dung CoboGesture gốc.

---

# 55. Luồng nghiên cứu hoàn chỉnh của đề tài

Nếu viết thành một sơ đồ duy nhất trong báo cáo, tôi đề xuất dùng:

```text
                         DATA COLLECTION
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
       RGB Video           Skeleton               IMU
       Camera              Pose                   Watch
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    TIMESTAMP SYNCHRONIZATION
                               │
                               ▼
                       DATA PREPROCESSING
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
         RGB Dataset      Skeleton Dataset    IMU Dataset
             │                 │                 │
             ▼                 ▼                 ▼
          VideoMAE           CTR-GCN          CNN-BiLSTM
             │                 │                 │
             ▼                 ▼                 ▼
           P_RGB             P_SK              P_IMU
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                       OUTPUT ALIGNMENT
                               │
                               ▼
                  ┌────────────────────────┐
                  │    FUSION EXPERIMENTS │
                  └────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       RGB + SK            RGB + IMU          SK + IMU
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
                     RGB + SK + IMU
                               │
                               ▼
                     PROBABILITY VOTING
                               │
                               ▼
                      TEMPORAL VOTING
                               │
                               ▼
                        POST-PROCESSING
                               │
                               ▼
                  CONTINUOUS ACTIVITIES
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
           Accuracy        Temporal IoU       Latency
           F1-score        Overlap            FPS
           Confusion       Boundary           GPU/CPU
           Matrix          accuracy           cost
```

---

## 56. Thứ tự triển khai thực tế tôi khuyên bạn

Đừng làm cả 3 model + fusion + realtime ngay từ đầu. Hãy đi theo thứ tự:

```text
STEP 01
Chuẩn hóa dataset + annotation
        ↓
STEP 02
Train RGB model
        ↓
STEP 03
Train Skeleton model
        ↓
STEP 04
Train IMU model
        ↓
STEP 05
Chuẩn hóa output 3 model
        ↓
STEP 06
Benchmark isolated
        ↓
STEP 07
Implement sliding-window continuous
        ↓
STEP 08
RGB-only continuous
        ↓
STEP 09
Skeleton-only continuous
        ↓
STEP 10
IMU-only continuous
        ↓
STEP 11
RGB + Skeleton voting
        ↓
STEP 12
RGB + IMU voting
        ↓
STEP 13
Skeleton + IMU voting
        ↓
STEP 14
RGB + Skeleton + IMU
        ↓
STEP 15
Temporal voting
        ↓
STEP 16
Post-processing
        ↓
STEP 17
Skeleton boundary detection
        ↓
STEP 18
Realtime benchmark
```

**Đây là cách triển khai an toàn nhất**, vì ở bất kỳ bước nào bạn cũng có một baseline chạy được để so sánh.

---

### Một điểm rất đáng chú ý từ CoboGesture

Bài báo cho thấy isolated recognition và continuous recognition phải được tách thành **hai tầng đánh giá**: isolated dùng accuracy, còn continuous dùng frame-wise accuracy và temporal action localization/overlap. Trong thí nghiệm CoboGesture, VideoMAE/VideoMAEv2 được train trên các đoạn isolated, sau đó mới đưa vào continuous pipeline.  

Vì vậy, với đề tài của bạn, **không nên train trực tiếp một model “RGB+Skeleton+IMU continuous” ngay từ đầu**. Hãy giữ kiến trúc:

> **3 isolated classifiers độc lập → common probability interface → multimodal voting → temporal processing.**

Đây sẽ là cấu trúc rõ ràng nhất để bạn vừa **kế thừa tư tưởng CoboGesture**, vừa có phần mở rộng nghiên cứu của riêng mình là **IMU + multimodal voting**.
