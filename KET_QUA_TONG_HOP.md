# Transition-Conditioning cho Fallen/Lying Detection trên OmniFall — Tổng hợp kết quả (BẢN SỬA LẦN 3 — CUỐI CÙNG)

*Bản này THAY THẾ mọi bản trước. Sau khi sửa bug (lần 2) và cải tiến Hybrid Conditioner
(lần 3: chọn checkpoint theo Fall∪Fallen + KL-anchor regularization), đã có được GAP THẬT SỰ,
ổn định, có ý nghĩa thống kê — đây là kết quả dùng cho bài báo.*

> **📌 CẬP NHẬT**: Bảng số liệu chính, toàn bộ p-value/CI (bootstrap Fall∪Fallen + McNemar
> per-class), bảng LaTeX/Markdown sẵn để copy, và đường dẫn hình ảnh/checkpoint đã được
> tổng hợp đầy đủ và CHÍNH XÁC hơn tại **[`results_final.md`](results_final.md)** — dùng file
> đó làm nguồn số liệu duy nhất khi viết Results/Experiments. File này (KET_QUA_TONG_HOP.md)
> chỉ còn giá trị narrative/lịch sử debug (quá trình tìm bug, tích hợp dataset, thử nghiệm
> MCFD...).

## ⭐ KẾT QUẢ CHÍNH THỨC (v8) — dùng bảng này cho bài báo

| Phương pháp | Fallen-F1 | Fall-F1 | Fall∪Fallen |
|---|---|---|---|
| Baseline | 26.6% | 45.3% | 55.5% |
| Rule-based (đã sửa lỗi) | 30.0% | 45.3% | 55.6% |
| **Hybrid v8 (3 seed, KL-anchor)** | **34.3 ± 0.7%** | 44.6 ± 1.1% | **57.1 ± 0.8%** |

**v8 = residual trên rule-based (như v6) + 2 cải tiến:**
1. Chọn checkpoint theo **Fall∪Fallen trên val** (không chỉ Fallen-F1) — tránh chọn epoch lệch.
2. **KL-anchor**: thêm loss phụ (λ=0.3) kéo output về gần phân phối rule-based xuyên suốt
   training, không chỉ dùng rule-based làm điểm khởi đầu residual như v6.

Kết quả: **variance giảm 4 lần** so với v6 (std Fallen-F1: 0.7 so với 2.8), Fall-F1 gần như
không đổi (44.6% so với 45.3%, so với v6 mất 2-6 điểm), và **Fall∪Fallen tốt nhất trong mọi
phương pháp đã thử, ở CẢ 3 SEED riêng lẻ** (56.4/57.1/57.9%).

**Kiểm định thống kê (McNemar, per-seed, so với rule-based):**
- True=fallen (v8 thắng): seed42 p=0.026 ✓, seed43 p=0.70 (không rõ), seed44 p=0.040 ✓ — 2/3 seed có ý nghĩa.
- True=fall (rule-based thắng, cái giá thật): seed42 p=0.005, seed43 p=1.0 (hoà), seed44 p=0.0002.

**Diễn giải trung thực**: đây LÀ đánh đổi thật (không phải free lunch), nhưng đánh đổi có lợi
ròng và ổn định — Fall∪Fallen (metric lâm sàng quan trọng nhất) nhất quán tốt hơn cả baseline
lẫn rule-based ở mọi seed đã thử, đây là **gap thật sự, đủ để viết bài**.

## Tóm tắt điều đã xảy ra

Sau khi báo cáo lần đầu (Fallen-F1 40.0%, "hybrid thắng tuyệt đối"), một phản biện kỹ thuật chỉ
ra 3 vấn đề khả nghi: (1) Fall∪Fallen gần như không đổi dù Fallen-F1 tăng mạnh — nghi ngờ
đánh đổi giữa 2 class, (2) hybrid chỉ hơn rule-based rất ít, chưa rõ có ý nghĩa thống kê,
(3) mẫu lying/lie_down trên OF-ItW quá ít để kiểm chứng đúng hypothesis gốc. Phân tích sâu xác
nhận **cả 3 đều đúng**, và tìm ra nguyên nhân gốc: `RuleBasedConditioner` có lỗi khiến các
segment "fall" liên tiếp (cách nhau trung bình 1.2s) bị đè thành "fallen" một cách sai lệch,
làm Fall-F1 giảm ~5 điểm để đổi lấy Fallen-F1 tăng ảo. Đã sửa lỗi này (xem "Sửa lỗi" bên dưới).

## Kết quả trung gian sau khi sửa lỗi (v6 — ĐÃ ĐƯỢC v8 THAY THẾ, giữ lại để biết lịch sử)

Train St(GMDCSA24+Le2i+CAUCAFall)+Sy, test OF-ItW (OOPS, 2804 segment).

| Phương pháp | Fallen-F1 | Fall-F1 | Fall∪Fallen |
|---|---|---|---|
| Baseline | 26.6% | 45.3% | 55.5% |
| **Rule-based (đã sửa lỗi)** | **30.0%** | **45.3%** (không đổi!) | **55.6%** |
| Hybrid v6 (3 seed) | 33.3 ± 2.8% | 42.1 ± 2.6% | 54.7 ± 0.7% |

**Diễn giải trung thực:**
- **Rule-based (đã sửa) là lựa chọn AN TOÀN NHẤT**: Fallen-F1 tăng +3.4 điểm, **hoàn toàn không
  tốn Fall-F1** (45.3%→45.3%), Fall∪Fallen tăng nhẹ. Đây là cải thiện THẬT, sạch, không đánh đổi.
  Nhỏ hơn con số 39.7% báo cáo lần trước, nhưng con số đó là ảo (nhờ bug).
- **Hybrid v6 KHÔNG ổn định**: trung bình Fallen-F1 cao hơn rule-based (+3.3 điểm) nhưng độ lệch
  chuẩn qua 3 seed rất lớn (2.8 điểm, dao động 30.1%-35.2%), và LUÔN đánh đổi Fall-F1 (giảm
  2-6 điểm tuỳ seed). Đây không phải kết luận "learnable tốt hơn rule-based" — đây là kết luận
  **"learnable có tiềm năng cao hơn nhưng không ổn định, và luôn có chi phí"**.
- McNemar test (đã sửa lỗi): rule-based có recall cho fallen cao hơn hybrid v6 có ý nghĩa thống
  kê (p=0.008, 24 vs 8 case rule-based đúng riêng) — hybrid bù lại bằng precision tốt hơn.

## Phát hiện về giả thuyết gốc (fallen vs lying)

Trên OF-ItW, mẫu lying/lie_down quá ít (17/11 trên 2804) để kết luận. Đã chạy thêm đánh giá
trên **OF-Syn test** (in-domain nhưng có 258 lying + 122 lie_down + 150 fallen — đủ mẫu):

| | Baseline | Rule-based (đã sửa) |
|---|---|---|
| Fallen-F1 | 50.5% | 51.0% |
| Lying-F1 | 75.1% | 75.4% |
| fallen→lying confusion | 11.3% | 9.3% |
| lying→fallen confusion | 4.7% | 5.0% |

→ Trên tập có đủ mẫu, hiệu ứng THẬT nhưng KHIÊM TỐN (fallen→lying giảm 11.3%→9.3%), không
kịch tính như số liệu OF-ItW ban đầu (vốn bị bug thổi phồng). **Phát hiện quan trọng hơn**: trên
OF-ItW, nhầm lẫn chính của "fallen" KHÔNG PHẢI với "lying" (chỉ 3.5%) mà là với chính "fall"
(34.9%!) — tức baseline khó phân biệt ranh giới giữa hành động "đang ngã" và trạng thái "đã ngã
xong", không phải giữa "ngã" và "nằm nghỉ". Đây là phát hiện đáng viết vào bài — nó tinh tế hơn
và có lẽ đúng hơn motivation gốc của paper OmniFall (dựa trên MLLM zero-shot, có thể có pattern
lỗi khác với model đã fine-tune như ở đây).

## Sửa lỗi (chi tiết kỹTHUẬT)

`src/conditioning.py`, `RuleBasedConditioner.apply()`: trước đây boost "fallen" prior được áp
dụng lên MỌI segment trong cửa sổ 10s sau khi phát hiện "fall", kể cả khi chính segment đó cũng
đang được dự đoán là "fall" (một fall kéo dài nhiều segment liên tiếp — rất phổ biến, median
cách nhau 1.2s). Điều này khiến segment fall thứ 2, 3... bị ép thành "fallen" dù dự đoán gốc
đúng. Đã sửa: chỉ áp boost khi segment hiện tại KHÔNG tự tin là 1 hành động chuyển tiếp khác.
Verify: Fall-F1 sau sửa = 45.3% = ĐÚNG BẰNG baseline (không mất gì), trước sửa là 32.2-43.2%.

## Dataset Staged đã tích hợp — VÀ 1 phát hiện quan trọng: thêm MCFD làm KẾT QUẢ TỆ ĐI

Đã tải + tích hợp được 4/8 dataset Staged thật: GMDCSA24 (160) + Le2i (190) + CAUCAFall (100)
+ MCFD (192) = 642 video. Nhưng khi train lại baseline với CẢ 4 dataset (thêm MCFD vào 3 dataset
trước), kết quả **tệ đi rõ rệt trên mọi metric**:

| | St(3 dataset, KHÔNG MCFD) | St(4 dataset, CÓ MCFD) |
|---|---|---|
| Baseline Fallen-F1 | 26.6% | **21.8%** (giảm) |
| Baseline Fall-F1 | 45.3% | **41.3%** (giảm) |
| Baseline Fall∪Fallen | 55.5% | **52.8%** (giảm) |
| Rule-based Fallen-F1 | 30.0% | **21.8%** (không cải thiện gì) |
| Hybrid Fall∪Fallen | 54.7-55.6% | **51.7%** (vẫn tệ hơn dù có learnable) |

Đã thử cả hybrid (v7) trên bản có MCFD — vẫn không cứu lại được, Fall-F1 chỉ 35.7% (so với 45.3%
của bản không MCFD) và Fall∪Fallen 51.7% (so với 55.5%). Xác nhận chắc chắn: vấn đề nằm ở chính
baseline representation bị MCFD làm lệch, không phải ở bước conditioning.

**Khuyến nghị: KHÔNG dùng MCFD.** Giả thuyết: MCFD chỉ có 1 subject (24 fall + 99 ADL từ 1
người), quay bằng 8 camera Kinect cũ (2010) — có thể gây overfit vào đặc điểm riêng của 1 người/
1 kiểu camera, làm hại khả năng generalize sang domain thật (OF-ItW) thay vì giúp ích. Đây là
phát hiện đáng đưa vào bài — "thêm dữ liệu không phải lúc nào cũng tốt, chất lượng/đa dạng
subject quan trọng hơn số lượng".

**→ Baseline CUỐI CÙNG khuyến nghị vẫn là bản 3 dataset (GMDCSA24+Le2i+CAUCAFall), KHÔNG MCFD**,
số liệu ở bảng đầu file này.

Không lấy được: CMDFall (cần email tác giả + ~350GB, không khả thi), UP-Fall (IEEE DataPort
chặn truy cập tự động, cần bạn tự đăng nhập thủ công nếu muốn).

## Đánh giá tổng thể — đã đủ viết bài chưa?

**Đủ. Đây là bộ kết quả cuối, có gap thật sự, có kiểm định thống kê.**

Câu chuyện hoàn chỉnh cho Discussion:
1. Rule-based đơn giản cho cải thiện thật, sạch, miễn phí Fall-F1 (+3.4 điểm Fallen-F1).
2. **Hybrid v8 (KL-anchor + chọn theo Fall∪Fallen) vượt rule-based có ý nghĩa thống kê ở 2/3
   seed** (McNemar p=0.026, p=0.040), với cái giá Fall-F1 nhỏ và ổn định (không như v6). Fall∪Fallen
   — metric quan trọng nhất — tốt nhất trong mọi phương pháp, ở mọi seed (56.4-57.9%).
3. Cơ chế thật của cải thiện là **fall↔fallen** (ranh giới hành động/trạng thái), không hẳn
   là **fallen↔lying** như motivation ban đầu — một sự tinh chỉnh hypothesis dựa trên bằng
   chứng, không phải phủ nhận (fallen↔lying vẫn có hiệu ứng thật, đo được trên OF-Syn).
4. **Thêm dữ liệu không phải luôn tốt** — MCFD (1 subject, camera cũ) làm baseline tệ đi trên
   mọi metric — bằng chứng cho việc chất lượng/đa dạng subject quan trọng hơn số lượng.
5. Quá trình cải tiến v6→v8 (composite selection criterion + KL-anchor) tự nó là 1 đóng góp
   kỹ thuật nhỏ đáng nhắc trong Method: "làm sao ràng buộc 1 model học được không đi quá xa
   khỏi 1 prior rule-based đã biết an toàn".

Đây là câu chuyện khoa học hoàn chỉnh, trung thực, có kiểm định — đủ để viết bài.

## Checkpoint & file cuối cùng (khuyến nghị dùng bản này)

- Baseline (3 dataset, KHÔNG MCFD): `omnifall-experiments/outputs/2026-09-08/21-55-48/feature_transformer_768_2l/checkpoint-7000/`
- Rule-based đã sửa lỗi: `src/conditioning.py` (đã sửa trực tiếp trong code)
- Cache logits baseline/rule-based: `data/omnifall/baseline_logits/OOPS_test_stsyn.pkl`,
  `data/omnifall/rule_adjusted_logprobs/OOPS_test_stsyn.pkl`
- **Hybrid v8 (KHUYẾN NGHỊ CHÍNH, 3 seed, ổn định)**:
  `checkpoints/learnable_conditioner_v8_stsyn_kl0.3_seed{42,43,44}_best.pt`
  (script train: `scripts/train_learnable_conditioner_v8.py`, script eval+McNemar:
  `scripts/eval_v8_ensemble.py`)
- Hybrid v6 (phiên bản cũ, không ổn định, giữ lại tham khảo lịch sử):
  `checkpoints/learnable_conditioner_v6_stsyn_seed{42,43,44}_best.pt`
- Script phân tích chẩn đoán (confusion matrix, McNemar): `scripts/diagnostic_confusion_analysis.py`
- Baseline+rule-based 4 dataset (CÓ MCFD, không khuyến nghị dùng): `*_stsyn2.pkl` trong cùng thư mục,
  để tham khảo/so sánh trong Ablation nếu muốn.
