# OmniFall Transition-Conditioning

Phát hiện trạng thái **fallen** (đã ngã, chưa dậy được) từ video trên benchmark
[OmniFall](https://arxiv.org/abs/2505.19889): kiểm toán và tiền xử lý dữ liệu, Transition-Conditioning,
đánh giá streaming và một ứng dụng giám sát realtime.

Nhóm: Nguyễn Duy Quân · Phạm Tuấn Vỹ · Nghiêm Thị Thu Huyền · Vũ Hoàng Anh — FPT University.

## Kết quả chính (OF-ItW test, 572 video, chạy streaming như camera thật)

| Hệ thống | Fallen-F1 | Fall-F1 | Fall∪Fallen-F1 |
| --- | --- | --- | --- |
| Baseline VideoMAE St+Sy | 31.9 | 45.5 | 56.8 |
| Fine-tune ItW-train | 40.8 | 60.4 | 67.7 |
| Fine-tune + TTA không gian + rule-conditioning | **44.3** | **62.0** | **69.3** |

Nhãn được so tại t − 2.2 s để bù độ trễ của cửa sổ nhân quả 4.4 s. Báo cáo đầy đủ:
[`deliverables/OmniFall_report.pdf`](deliverables/OmniFall_report.pdf); slide:
[`deliverables/OmniFall_slides.html`](deliverables/OmniFall_slides.html).

Những phát hiện về dữ liệu:

- 154/818 video OOPS 23.98 fps bị lấy mẫu ở 8 fps thay vì 7.5 fps, lệch nhãn tới 2.2 s (đã sửa: lấy mẫu theo đồng hồ).
- Center-crop chỉ giữ trung vị 56% khung hình (đã thêm TTA không gian 3 crop).
- Đánh giá với vị trí crop ngẫu nhiên làm baseline dao động ±0.8 điểm (đã cố định vị trí).
- 92.3% cặp segment liền kề ở OF-ItW khác nhãn, nên đánh giá theo segment cho mô hình biết trước chỗ nhãn đổi; streaming là thước đo chính.

## Cấu trúc

```
tcfall/          thư viện: config, data, video, features, preprocess_gpu, segments,
                 baseline, conditioners, realtime, metrics, study
pipeline/        s01 → s08: kiểm toán, trích feature, thí nghiệm, đánh giá, tổng hợp
app/             realtime_app.py — ứng dụng giám sát 3 luồng (webcam / video)
deliverables/    báo cáo PDF, slide HTML, build.py để dựng lại cả hai
outputs_v2/      bảng, hình và cấu hình đã chọn (best_config.json)
checkpoints/     conditioner Hybrid v8 (3 seed) và các phiên bản trước
src/, scripts/   code của phiên bản trước, giữ để đối chiếu với bài báo
third_party/     patch cho repo omnifall-experiments chính thức
```

## Cài đặt

```bash
pip install -r requirements.txt
git clone https://github.com/simplexsigil/omnifall-experiments.git
cd omnifall-experiments && git checkout af4338146c81237d91c3b6d2d912ca7bd4969cc1
git apply ../third_party/omnifall-experiments/changes.patch
cp -r ../third_party/omnifall-experiments/configs/* configs/
```

Dữ liệu tải từ Hugging Face `simplexsigil/omnifall` (nhãn, split, OF-Syn) và từ nguồn gốc của từng bộ
staged (GMDCSA24, Le2i, CAUCAFall); video OF-ItW chuẩn bị bằng `prepare_oops_videos.py` của bộ dữ liệu.
Đặt tất cả trong `data/omnifall/`. Trọng số baseline (checkpoint-7000) và classifier fine-tune không nằm
trong repo vì dung lượng; huấn luyện lại bằng omnifall-experiments và `pipeline/s05b_finetune_itw.py`.

Trên Windows nên đặt `PYTHONUTF8=1`.

## Chạy lại toàn bộ

```bash
python pipeline/s01_audit_data.py
python pipeline/s01b_video_audit.py
python pipeline/s02_extract_oops_features.py
python pipeline/s02b_extract_spatial_crops.py
python pipeline/s03_cache_logits.py
python pipeline/s03_cache_logits.py --variants timefix_start,timefix_end
python pipeline/s04_posthoc_study.py
python pipeline/s05_train_hybrid.py --paper_setup --tag paper --data_modes stsyn --lambdas 0,0.3,1.0
python pipeline/s05b_finetune_itw.py
python pipeline/s06_check_gpu_preprocess.py
python pipeline/s07_stream_eval.py
TCFALL_STREAM_TAG=ft TCFALL_BASELINE_CKPT=outputs_v2/finetune/baseline_ft_itw_seed42.pt python pipeline/s07_stream_eval.py
python pipeline/s08_final_report.py
python deliverables/build.py
```

## Ứng dụng realtime

```bash
python app/realtime_app.py                       # webcam 0, cấu hình tốt nhất
python app/realtime_app.py --video <clip>.mp4 --record out.mp4
python app/realtime_app.py --model base          # checkpoint St+Sy gốc
python app/realtime_app.py --alarm_on 1.5 --beep
```

Phím: `1` baseline · `2` rule · `3` hybrid · `4` hmm · `r` reset · `s` chụp màn hình · `space` tạm dừng ·
`q` thoát. Cảnh báo được ghi vào `outputs_v2/app_logs/alarm_events.csv`. Trên RTX 3070: 40–47 ms mỗi cửa
sổ, kể cả TTA 3 crop.

## Giới hạn

- Backbone chưa được huấn luyện lại với feature đã sửa lấy mẫu (feature OF-Syn và staged cần trích lại).
- Chỉ 3/8 bộ staged; val OF-ItW nhỏ (81 video).
- Báo động giả đo trên clip "fail" ngắn, chưa phải camera giám sát liên tục.
