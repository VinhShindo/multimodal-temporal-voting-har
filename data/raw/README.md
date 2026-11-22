# Cấu trúc dữ liệu raw

Kiểm kê được thực hiện ngày 2026-11-14. Số lượng dưới đây mô tả các file đang
có trên máy; không khẳng định mọi tổ hợp subject/camera/scene/session đã đầy
đủ.

```text
data/raw/
├── trimmed/
│   ├── annotation/
│   ├── cross_scene_video/
│   ├── cross_view_video/
│   ├── trimmed_pose/pose/{cross_scene,cross_view}/trainval/
│   ├── trimmed_sensor/sensor/
│   │   ├── acc_phone_clip/
│   │   ├── acc_watch_clip/
│   │   ├── gyro_clip/
│   │   └── orientation_clip/
│   └── trimmed_splits/splits/
└── untrimmed/
    ├── annotation/trainval/
    ├── sensor/trainval/{acc_phone_clip,acc_watch_clip,gyro_clip,orientation_clip}/
    ├── splits/
    └── video/trainval/{cam1,cam2,cam3}/
```

## Dữ liệu trimmed

| Thư mục | Nội dung | Số file |
|---|---|---:|
| `trimmed/annotation/` | Ground-truth JSON cho test scene và view | 2 |
| `trimmed/cross_scene_video/` | Video clip đã cắt theo cross-scene | 17.820 |
| `trimmed/cross_view_video/` | Video clip đã cắt theo cross-view | 21.174 |
| `trimmed/trimmed_pose/pose/` | Pose JSON thuộc hai cây `cross_scene`, `cross_view` | 38.994 |
| `trimmed/trimmed_sensor/sensor/` | IMU CSV clip thuộc bốn loại sensor | 35.685 |
| `trimmed/trimmed_splits/splits/` | Train/val split cho x-scene và x-view | 4 |

Ví dụ đường dẫn:

```text
trimmed/cross_view_video/trimmed_x-view_subjectN/video/cross_view/trainval/subjectN/camN/sceneN/sessionN/action.mp4
trimmed/trimmed_pose/pose/cross_view/trainval/subjectN/camN/sceneN/sessionN/action.json
trimmed/trimmed_sensor/sensor/acc_watch_clip/subjectN/sceneN/sessionN/action.csv
```

Pose có cả cây `cross_scene` và `cross_view`. Đường dẫn IMU không có camera
vì một sự kiện sensor có thể dùng chung giữa nhiều góc quay. Dữ liệu hiện có
các định danh thuộc 20 subject, cam1–4, scene1–4, session1–5; không có nghĩa
là mọi tổ hợp đều tồn tại. Sensor mặc định của bước preprocessing/EDA là
`acc_watch_clip`.

## Dữ liệu untrimmed

| Thư mục | Nội dung | Số file |
|---|---|---:|
| `untrimmed/video/` | Video ghi hình nguyên session | 706 |
| `untrimmed/sensor/` | IMU CSV nguyên session thuộc bốn loại sensor | 11.573 |
| `untrimmed/annotation/` | Annotation text theo session | 240 |
| `untrimmed/splits/` | Danh sách train/val cho x-session | 2 |

Video theo camera: cam1 có 234 file, cam2 có 238, cam3 có 234. Sensor CSV:
3.290 `acc_phone_clip`, 3.242 `acc_watch_clip`, 3.290 `gyro_clip`, và 1.751
`orientation_clip`. Dữ liệu hiện có các định danh thuộc 20 subject, cam1–3,
scene1–4, session1–3; không khẳng định mọi tổ hợp đã đầy đủ.

Ví dụ đường dẫn:

```text
untrimmed/video/trainval/camN/subjectN/sceneN/sessionN/timestamp.mp4
untrimmed/sensor/trainval/sensor_type/subjectN/sceneN/sessionN/timestamp.csv
untrimmed/annotation/trainval/subjectN/sceneN/sessionN/subjectN_sceneN_sessionN.txt
untrimmed/splits/untrimmed_x-session_{train,val}.txt
```

Mỗi dòng annotation mẫu ghi khoảng thời gian bắt đầu-kết thúc và nhãn hành
động. File sensor mẫu `acc_phone_clip` có header `time,x,y,z`; một số CSV
trimmed `acc_watch_clip` được kiểm tra không có header và chứa timestamp cùng
ba giá trị số. Định dạng có thể khác theo loại sensor.

Các thư mục untrimmed đã được giải nén; trong lần kiểm kê này không tìm thấy
file archive `.zip` nào bên dưới `data/raw`.

## Preprocessing và phạm vi EDA

Không chỉnh sửa trực tiếp dữ liệu raw. Dữ liệu trimmed dạng action-level
được dùng để tạo sáu manifest train/val tại
`data/processed/MMAct/manifests/protocols/`. EDA clip hiện tại đọc các
manifest này; duplicate `clip_id` giữa hai cây nguồn được gộp khi tạo
protocol. Vì vậy, 27.527 clip RGB/Skeleton trong EDA không bằng tổng 38.994
file video của hai cây cross-scene và cross-view. EDA clip chưa phân tích
annotation hoặc đặc trưng theo session của dữ liệu untrimmed. Split x-session
untrimmed độc lập với split x-scene/x-view trimmed.
