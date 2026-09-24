# README — Nhận dạng hoạt động con người đa phương thức bằng Multi-Modal Temporal Voting

## 1. Tổng quan

Hệ thống nhận dạng hoạt động con người từ ba nguồn dữ liệu:

- **RGB** từ camera.
- **Skeleton** từ camera thông qua pose estimation.
- **IMU** từ smartwatch.

Mục tiêu:

- Huấn luyện từng modality model độc lập.
- Kết hợp dự đoán từ nhiều modality bằng voting.
- Làm mượt và hậu xử lý theo thời gian để tạo chuỗi hành động liên tục.
- Đánh giá cả nhận dạng isolated và continuous.

---

## 2. Kiến trúc tổng thể

![Kiến trúc tổng thể](<docs/Kiến trúc tổng quan.png>)

---

## 3. Nguồn dữ liệu

![Collect Data](<docs/Collect Data.png>)

### 3.1 Camera

Camera cung cấp hai luồng:

- **RGB video**: chuỗi frame màu.
- **Skeleton sequence**: chuỗi khung xương được trích xuất từ pose estimation.

### 3.2 Smartwatch

Smartwatch cung cấp tín hiệu IMU:

- Accelerometer.
- Gyroscope.
- Orientation.

### 3.3 Đồng bộ

Dữ liệu từ camera và smartwatch được đồng bộ theo timestamp.

---

## 4. Tiền xử lý

### 4.1 RGB

- Resize frame.
- Chuẩn hóa giá trị pixel.
- Cắt thành các clip theo cửa sổ thời gian.
- Có thể lấy mẫu frame với stride cố định.

### 4.2 Skeleton

- Chuẩn hóa tọa độ khớp.
- Xử lý missing joint bằng nội suy.
- Căn chỉnh theo hip center hoặc root joint.
- Chuẩn hóa theo chiều cao cơ thể.

### 4.3 IMU

- Resample về tần số cố định.
- Lọc nhiễu.
- Chuẩn hóa từng kênh.
- Cắt theo cùng cửa sổ thời gian với RGB và Skeleton.

### 4.4 Sliding Window

```text
Window length: L
Stride:        S
```

Ví dụ:

```text
Window 1: [t0, t0 + L]
Window 2: [t0 + S, t0 + S + L]
Window 3: [t0 + 2S, t0 + 2S + L]
...
```

---

## 5. Mô hình nhận dạng từng modality

![Train Model Architecture](<docs/Train Model Architecture.png>)

### 5.1 IMU Model

Ví dụ output:

```text
walking      0.72
standing     0.12
drinking     0.08
sitting      0.05
...
```

### 5.2 RGB Model

Ví dụ output:

```text
drinking     0.81
walking      0.08
standing     0.04
...
```

### 5.3 Skeleton Model

Ví dụ output:

```text
drinking     0.69
standing     0.14
walking      0.09
...
```

---

## 6. Continuous Recognition Pipeline

![Realtime Architecture](<docs/Realtime Architecture.png>)

Luồng xử lý:

1. Đồng bộ dữ liệu camera và smartwatch.
2. Chia chuỗi liên tục thành các cửa sổ.
3. Mỗi modality model dự đoán trên từng cửa sổ.
4. Kết hợp dự đoán giữa các modality bằng voting.
5. Kết hợp dự đoán theo thời gian bằng temporal voting.
6. Hậu xử lý để loại bỏ nhiễu và nối đoạn.
7. Xuất chuỗi hành động cuối cùng.

---

## 7. Modality Voting

### 7.1 Majority Voting

Mỗi modality bỏ một phiếu.

```text
RGB       → drinking
Skeleton  → drinking
IMU       → walking

→ drinking
```

Công thức:

```text
y = mode(argmax P(action | modality))
```

### 7.2 Probability Voting

Dùng xác suất của từng model thay vì chỉ một phiếu.

```text
                    DRINKING

RGB       = 0.82
Skeleton  = 0.71
IMU       = 0.55

Tổng = 0.82 + 0.71 + 0.55 = 2.08
```

```text
                    WALKING

RGB       = 0.04
Skeleton  = 0.12
IMU       = 0.35

Tổng = 0.04 + 0.12 + 0.35 = 0.51
```

Chọn class có tổng xác suất cao nhất.

Công thức:

```text
S(c) = Σ_m P(c | modality_m)
y = argmax_c S(c)
```

### 7.3 Weighted Probability Voting

Mỗi modality có trọng số riêng.

```text
RGB       × 0.4
Skeleton  × 0.3
IMU       × 0.3
```

Công thức:

```text
S(c) = Σ_m w_m * P(c | modality_m)
y = argmax_c S(c)
```

Trong đó:

```text
Σ_m w_m = 1
```

Trọng số được xác định trước hoặc tối ưu trên tập validation.

---

## 8. Temporal Voting và Post-processing

### 8.1 Temporal Voting

Sau khi có dự đoán trên từng cửa sổ:

```text
Window 1:
RGB       → standing
Skeleton  → standing
IMU       → standing

Window 2:
RGB       → standing
Skeleton  → drinking
IMU       → drinking

Window 3:
RGB       → drinking
Skeleton  → drinking
IMU       → drinking

Window 4:
RGB       → drinking
Skeleton  → drinking
IMU       → drinking

Window 5:
RGB       → drinking
Skeleton  → drinking
IMU       → drinking
```

Temporal voting làm mượt để tạo:

```text
standing ─────────→ drinking
```

và xác định:

```text
START = Window 3
END   = Window 5
```

### 8.2 Post-processing

Các bước hậu xử lý:

1. Làm mượt chuỗi dự đoán:
   - Moving average.
   - Median filter.
   - Majority filter theo cửa sổ trượt.

2. Loại bỏ đoạn ngắn:
   - Nếu một hành động xuất hiện dưới `min_duration`, loại bỏ hoặc gộp vào đoạn lân cận.

3. Ngưỡng confidence:
   - Nếu confidence < `threshold`, đánh dấu là unknown hoặc giữ nhãn trước đó.

4. Gộp đoạn:
   - Hai đoạn cùng nhãn nằm gần nhau được gộp thành một.

5. Xác định boundary:
   - Có thể dùng Skeleton motion change để xác định START/END.
   - Hoặc dùng thay đổi confidence giữa các cửa sổ.

---

## 9. Kiến trúc mở rộng với Skeleton Boundary

Skeleton có thể được dùng để xác định boundary hành động.

Luồng:

1. Skeleton được phân tích để phát hiện thay đổi chuyển động.
2. Xác định khoảng START — END của hành động.
3. Trong khoảng đó, RGB và IMU được đưa vào classifier.
4. Voting và temporal post-processing tạo hành động cuối cùng.

---

## 10. Cấu hình thí nghiệm

### 10.1 Thí nghiệm đơn modality

| Cấu hình | Mô tả |
|---|---|
| IMU | Chỉ dùng smartwatch |
| RGB | Chỉ dùng camera RGB |
| Skeleton | Chỉ dùng skeleton |

### 10.2 Thí nghiệm hai modality

| Cấu hình | Mô tả |
|---|---|
| IMU + RGB | Smartwatch + RGB |
| IMU + Skeleton | Smartwatch + Skeleton |
| RGB + Skeleton | RGB + Skeleton |

### 10.3 Thí nghiệm ba modality

| Cấu hình | Mô tả |
|---|---|
| RGB + Skeleton + IMU | Cả ba nguồn |

### 10.4 Thí nghiệm voting

| Cấu hình | Mô tả |
|---|---|
| Majority Voting | Mỗi modality một phiếu |
| Probability Voting | Cộng xác suất |
| Weighted Probability Voting | Cộng xác suất có trọng số |

### 10.5 Thí nghiệm temporal

| Cấu hình | Mô tả |
|---|---|
| Không temporal | Dự đoán từng window độc lập |
| Moving Average | Làm mượt trung bình động |
| Median Filter | Làm mượt trung vị |
| Threshold + Merge | Ngưỡng confidence + gộp đoạn |
| Skeleton Boundary | Dùng skeleton xác định boundary |

---

## 11. Đánh giá

### 11.1 Isolated Recognition

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

### 11.2 Continuous Recognition

- Frame-wise accuracy.
- Segment-wise accuracy.
- Activity overlap score.
- Edit distance.
- Latency.

### 11.3 Bảng kết quả đề xuất

| Cấu hình | Accuracy | F1 |
|---|---:|---:|
| IMU | ? | ? |
| RGB | ? | ? |
| Skeleton | ? | ? |
| IMU + RGB — Majority | ? | ? |
| IMU + Skeleton — Majority | ? | ? |
| RGB + Skeleton — Majority | ? | ? |
| 3 Modalities — Majority | ? | ? |
| 3 Modalities — Probability | ? | ? |
| 3 Modalities — Weighted Probability | ? | ? |

| Cấu hình | Frame Accuracy | Segment/Action Score |
|---|---:|---:|
| RGB | ? | ? |
| RGB + Skeleton | ? | ? |
| RGB + Skeleton + IMU | ? | ? |

---

## 12. Định dạng đầu ra

```json
{
  "start": 12.5,
  "end": 15.8,
  "action": "drinking",
  "confidence": 0.87,
  "modality_scores": {
    "RGB": 0.82,
    "Skeleton": 0.71,
    "IMU": 0.55
  }
}
```

---

## 13. Pseudocode

```python
# Training
for modality in ["RGB", "Skeleton", "IMU"]:
    train_model(modality)

# Inference
streams = synchronize(camera, smartwatch)
windows = sliding_window(streams, length=L, stride=S)

window_predictions = []

for window in windows:
    probs = {}

    for modality in ["RGB", "Skeleton", "IMU"]:
        probs[modality] = model[modality].predict(window[modality])

    fused = modality_voting(probs)
    window_predictions.append(fused)

final_sequence = temporal_voting(window_predictions)
final_sequence = post_process(final_sequence)

return final_sequence
```

---

## 14. Cấu trúc thư mục

```text
project/
├── data/
│   ├── rgb/
│   ├── skeleton/
│   └── imu/
├── models/
│   ├── rgb_model.py
│   ├── skeleton_model.py
│   └── imu_model.py
├── fusion/
│   ├── modality_voting.py
│   ├── temporal_voting.py
│   └── post_processing.py
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── README.md
```

---

## 15. Tóm tắt luồng hệ thống

Hệ thống huấn luyện từng modality model độc lập, sau đó kết hợp dự đoán trên các cửa sổ thời gian bằng modality voting và temporal voting. Kết quả cuối cùng là chuỗi hành động liên tục sau hậu xử lý.