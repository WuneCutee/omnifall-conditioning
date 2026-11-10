# Notebooks giải thích và chạy thật

Mỗi notebook đi qua một bước của pipeline. Phần đầu là code kèm giải thích, phần **▶ Chạy thật** ở cuối dùng dữ liệu, checkpoint và GPU của dự án. Kết quả đã được lưu sẵn trong từng notebook, mở lên là xem được.

| Notebook | Nội dung | Phần chạy thật |
|---|---|---|
| `00_tong_quan_va_cau_hinh` | Bản đồ pipeline, cấu hình chung | Đọc cấu hình đã chọn, kiểm tra dữ liệu |
| `01_tien_xu_ly_du_lieu` | Nạp nhãn, gán split, làm sạch, kiểm toán | Kiểm toán 5 bộ dữ liệu |
| `02_lay_mau_video_va_trich_dac_trung` | Lấy mẫu 7.5 fps, tiền xử lý, trích feature | Đo độ trôi, trích feature và so với file đã lưu |
| `03_cat_doan_va_cache_logits` | Chọn 18 feature cho mỗi đoạn | Dự đoán từng đoạn của một video thật |
| **`04_backbone_mo_hinh`** | **Toàn bộ slide 11, viết đầy đủ trong notebook** | **12 lớp VideoMAE và 2 lớp Encoder viết tay, nạp trọng số thật, chạy trên video thật** |
| `05_finetune_itw` | Fine-tune classifier trên ItW-train | Huấn luyện 2 epoch, chấm lại checkpoint đã lưu |
| `06_luat_chuyen_trang_thai` | Quy tắc, HMM, Hybrid | So các cách làm mượt trên val |
| `07_danh_gia_streaming` | Metric, McNemar, đánh giá streaming | Tái lập bảng kết quả trên tập test |
| `08_realtime_engine` | Engine của web demo | Chạy từng ảnh trên video thật |

## Muốn chạy lại

Mở notebook từ thư mục `notebooks/` và chạy từ trên xuống. Cần:

- `torch`, `transformers`, `opencv-python`, `matplotlib`, `pandas`. VideoMAE tự tải từ HuggingFace.
- Dữ liệu trong `data/` và kết quả trong `outputs_v2/` theo README chính. Các thư mục này và file `.pt` không nằm trong repo vì dung lượng.
- Repo `omnifall-experiments` theo README chính, trừ notebook 04 vốn tự chứa kiến trúc. Notebook 04 chỉ cần checkpoint `outputs_v2/finetune/baseline_ft_itw_seed42.pt` và một video, đặt qua biến môi trường `FALL_VIDEO` nếu không có dữ liệu OOPS.
