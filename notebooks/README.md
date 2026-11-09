# Notebooks giải thích code

Mỗi notebook đi qua một bước của pipeline. Cell code được chép nguyên văn từ file nguồn trong repo, cell markdown phía trên giải thích đoạn code đó làm gì và vì sao.

| Notebook | Nội dung |
|---|---|
| `00_tong_quan_va_cau_hinh` | Bản đồ pipeline, cấu hình chung (`tcfall/config.py`) |
| `01_tien_xu_ly_du_lieu` | Nạp nhãn, gán split, làm sạch, kiểm toán dữ liệu |
| `02_lay_mau_video_va_trich_dac_trung` | Lấy mẫu 7.5 fps (lỗi trôi), tiền xử lý ảnh, trích feature VideoMAE |
| `03_cat_doan_va_cache_logits` | Chọn 18 feature cho mỗi segment (lỗi crop), cache logits |
| `04_backbone_mo_hinh` | Kiến trúc FeatureTransformer, nạp checkpoint |
| `05_finetune_itw` | Fine-tune classifier trên ItW-train (L2-SP, tăng cường feature) |
| `06_luat_chuyen_trang_thai` | Luật chuyển trạng thái, HMM, prior shift, Hybrid |
| `07_danh_gia_streaming` | Metric, McNemar, bootstrap, đánh giá streaming |
| `08_realtime_engine` | Engine suy luận thời gian thực cho app |

Mở notebook từ thư mục `notebooks/` (cell đầu tự thêm thư mục gốc repo vào `sys.path`). Các cell định nghĩa hàm và các cell **Ví dụ nhỏ** chạy được không cần dữ liệu; các hàm `main()` cần dữ liệu trong `data/` và `outputs_v2/` theo README chính.
