# `inference/` và `postprocess/` — Thuộc Giai đoạn 3 (Continuous Recognition)

## 1. Vị trí trong pipeline

```
GIAI ĐOẠN 1 ✅ — Train isolated models
  Skeleton (0.58)  +  IMU (0.98)  +  RGB (chưa)

GIAI ĐOẠN 2 ✅ — Fusion voting (đã code xong, chưa chạy)
  7 tổ hợp × 3 methods → best fused prediction

GIAI ĐOẠN 3 ⏳ — Continuous Recognition ← inference/ + postprocess/ Ở ĐÂY
  Video dài (untrimmed) → window → predict → clean → segments

GIAI ĐOẠN 4 — Self-collected data
GIAI ĐOẠN 5 — Real-time deployment
```

**Cả 2 folder chỉ dùng khi chạy trên video liên tục**, không phải trên clip đã cắt.

---

## 2. `inference/` — Chức năng

| File | Vai trò | Input | Output |
|---|---|---|---|
| `sliding_window.py` | Sinh cửa sổ trượt từ video dài | Tổng duration | list `(start, end)` |
| `boundary.py` | Phát hiện start/end của action từ skeleton motion | Skeleton sequence | list boundary timestamps |
| `predictor.py` | Wrapper: model + preprocess → prediction | Raw input + window | `Prediction` object |
| `isolated.py` | Chạy model trên nhiều clip isolated | Model + loader | list `Prediction` |

**Ví dụ usage:**
```python
# Video dài 5 phút
windows = generate_windows(total_duration=300, size=2.0, step=0.5)
# → [(0, 2), (0.5, 2.5), (1.0, 3.0), ...]

for w in windows:
    pred = predictor.predict(video_segment(w), w.id, w.start, w.end)
    # pred: {class_id=3, confidence=0.82, ...}
```

**Điều kiện cần:**
- ✅ Model đã train (skeleton/imu có)
- ✅ Config `configs/inference/sliding_window.yaml` (đã có)
- ❌ **Data untrimmed** (video dài chưa cắt) — CHƯA TẢI

---

## 3. `postprocess/` — Chức năng

Sau khi inference sinh ra hàng trăm predictions nhiễu, postprocess làm sạch:

| File | Vai trò | Trước | Sau |
|---|---|---|---|
| `smoothing.py` | Làm mượt theo lân cận thời gian | `[walk, walk, run, walk, walk]` | `[walk, walk, walk, walk, walk]` |
| `linking.py` | Nối segment cùng nhãn nếu gap < t_avg | `[walk 0-2], [walk 3-5]` | `[walk 0-5]` |
| `removing.py` | Loại segment quá ngắn | `[walk 4s], [run 0.1s], [walk 3s]` | `[walk 4s], [walk 3s]` |
| `pipeline.py` | Chạy tuần tự 3 bước trên | raw segments | clean segments |

**Ví dụ pipeline:**
```python
raw_preds = [pred1, pred2, ..., pred100]   # từ inference
clean_segments = run_postprocess(raw_preds)
# → [Segment(walk, 0.0, 12.5), Segment(run, 12.5, 18.0), ...]
```

**Điều kiện cần:**
- ✅ Code đã có
- ✅ Config `configs/postprocess/*.yaml` (đã có)
- ❌ **Raw predictions từ continuous inference** — CHƯA CÓ

---

## 4. Toàn bộ flow Giai đoạn 3

```
UNTRMMED VIDEO (5 phút)
       │
       ▼
┌─────────────────────────────────────────────────┐
│ inference/                                      │
│                                                  │
│  1. Load video → sample frames                   │
│  2. sliding_window.py → sinh windows            │
│  3. Với mỗi window:                              │
│     - Extract skeleton (MediaPipe)              │
│     - Extract IMU (từ file CSV)                 │
│     - predictor.py: chạy 3 models → probs       │
│     - fuse_probs → final class                  │
│  4. (Optional) boundary.py: detect start/end    │
└─────────────────────────────────────────────────┘
       │
       │  Output: [Prediction(window_1), ..., Prediction(window_N)]
       ▼
┌─────────────────────────────────────────────────┐
│ postprocess/                                     │
│                                                  │
│  1. smoothing.py  → làm mượt probs               │
│  2. Group thành segments                         │
│  3. linking.py    → nối segment cùng nhãn        │
│  4. removing.py   → loại segment quá ngắn        │
└─────────────────────────────────────────────────┘
       │
       ▼
FINAL SEGMENTS
[walk 0-12s], [run 12-18s], [sitting 18-25s], ...
       │
       ▼
Evaluate: frame-wise accuracy, temporal IoU
```

---

## 5. Điều kiện để chạy Giai đoạn 3

| Điều kiện | Trạng thái | Cần làm gì |
|---|---|---|
| Model isolated | ✅ SK + IMU | Chờ RGB |
| Fusion voting | ✅ Code xong | Chạy `40_benchmark_fusion.py` |
| **Data untrimmed** | ❌ | **Tải `untrimmed train+val` (~100 GB)** |
| Annotation untrimmed | ❌ | Nằm trong bộ untrimmed |
| MediaPipe skeleton extraction | ❌ | Viết `tools/extract_skeleton.py` |
| `inference/*` code | ⚠️ Placeholder | Viết đầy đủ |
| `postprocess/*` code | ⚠️ Placeholder | Viết đầy đủ |

---

## 6. Vậy `inference/` và `postprocess/` được viết ở bước nào?

**Thứ tự chính xác:**

```
BƯỚC 1 — Chạy `40_benchmark_fusion.py` (ngay bây giờ)
   └─ Xác nhận fused F1 ≥ 0.98

BƯỚC 2 — Tải untrimmed data (~100 GB, 1-2 ngày)
   └─ untrimmed video + sensor + annotation

BƯỚC 3 — Viết code inference (~1 ngày)
   ├─ extract_skeleton.py (MediaPipe)
   ├─ src/inference/sliding_window.py       (hoàn thiện)
   ├─ src/inference/predictor.py            (hoàn thiện)
   ├─ src/inference/boundary.py             (hoàn thiện)
   └─ scripts/50_run_continuous.py          (mới)

BƯỚC 4 — Viết code postprocess (~4 giờ)
   ├─ src/postprocess/smoothing.py          (hoàn thiện)
   ├─ src/postprocess/linking.py            (hoàn thiện)
   ├─ src/postprocess/removing.py           (hoàn thiện)
   └─ src/postprocess/pipeline.py           (hoàn thiện)

BƯỚC 5 — Chạy continuous inference
   └─ scripts/21_eval_continuous.py

BƯỚC 6 — Đo metrics
   ├─ Frame-wise accuracy
   ├─ Temporal IoU
   └─ Boundary accuracy
```

**Tóm lại:**
- `inference/` → **Bước 3** (sau khi tải untrimmed data)
- `postprocess/` → **Bước 4** (ngay sau inference)

---

## 7. Timeline ước tính

| Bước | Thời gian code | Thời gian chạy | Phụ thuộc |
|---|---|---|---|
| Chạy fusion | 0 | 5 phút | Có sẵn |
| Tải untrimmed | 0 | **1-2 ngày** | Mạng |
| MediaPipe extraction | 4h | 2-3h/50 video | Data |
| `inference/*` | 6h | — | Code |
| `postprocess/*` | 4h | — | Code |
| Chạy & đánh giá | 1h | 2-3h | Code |

→ **Tổng: ~3-4 ngày** (bao gồm chờ tải).

---

## 8. Câu hỏi bạn nên tự hỏi

**Câu 1: Bạn có cần Continuous Recognition không?**

- **Có** → Cần tải untrimmed + viết inference/postprocess. Đây là phần **thể hiện tinh thần CoboGesture** trong đồ án.
- **Không** → Chỉ cần isolated + fusion là đủ. Bỏ qua Giai đoạn 3.

**Câu 2: Nếu có, khi nào làm?**

- **Ngay sau khi fusion chạy xong** — vì fusion là nền tảng cho continuous.
- **Hoặc sau khi tự thu data** — vì bạn có untrimmed video tự quay (dễ hơn MMAct untrimmed).

**Câu 3: Có alternative nào rẻ hơn không?**

Thay vì tải MMAct untrimmed 100 GB, dùng **video tự thu** của bạn:
- Bạn có `collector.py` ghi video liên tục (không cắt) → đã là untrimmed
- Annotation được lưu với `start_time/end_time` → đã là ground truth
- Chỉ cần viết inference/postprocess là chạy được

→ **Đây là hướng tiết kiệm nhất** — vừa có data tự thu, vừa test continuous.

---

## 9. Đề xuất

**Hướng A — Rẻ, nhanh (khuyến nghị):**
```
1. Chạy fusion (ngay bây giờ)
2. Thu data tự thu (đã có pipeline ESP32)
3. Viết inference/postprocess → chạy trên video tự thu
4. Báo cáo: MMAct (isolated) + Self-collected (continuous)
```

**Hướng B — Đầy đủ, nặng:**
```
1. Chạy fusion
2. Tải MMAct untrimmed (~100 GB)
3. Viết inference/postprocess → chạy trên MMAct untrimmed
4. Báo cáo: MMAct (isolated + continuous) + Self-collected
```

**Hướng C — MVP:**
```
1. Chạy fusion
2. Bỏ qua continuous → kết thúc đồ án ở isolated + fusion
3. Báo cáo: Fusion 3-modal đạt F1 = X
```

---

## Trả lời trực tiếp

> `postprocess/` và `inference/` làm ở bước nào?

**Bước tiếp theo sau Fusion** — thuộc Giai đoạn 3 (Continuous Recognition).

**Cần chuẩn bị trước:**
1. Chạy `40_benchmark_fusion.py` xác nhận fusion hoạt động
2. Tải untrimmed data **hoặc** dùng video tự thu
3. Viết `extract_skeleton.py` (MediaPipe)

**Bạn muốn làm theo hướng nào?**
- **A.** Rẻ — dùng video tự thu (tôi viết code continuous cho video của bạn)
- **B.** Đầy đủ — tải MMAct untrimmed
- **C.** Bỏ qua continuous, chỉ isolated + fusion

Chọn hướng để tôi viết code tiếp theo.