# Results Final — Nguồn số liệu chính thức duy nhất

> File này là nguồn DUY NHẤT dùng để viết Results/Experiments. Không dùng số liệu từ các
> phiên bản v6/v7 cũ (đã lỗi thời / có bug đã sửa) hay từ KET_QUA_TONG_HOP.md cho các con số —
> file đó chỉ giữ vai trò narrative/lịch sử debug, số liệu tham chiếu phải lấy ở đây.
> Toàn bộ số liệu bên dưới được tính trên **OF-ItW test set (OOPS_test_stsyn), n=2804 segment,
> 572 video**, baseline St+Sy (checkpoint `feature_transformer_768_2l/checkpoint-7000`).

Ngày tổng hợp: 2026-09-09

---

## 1. Bảng kết quả chính (copy thẳng vào bài)

### 1a. Markdown

| Method | Fallen-F1 (%) | Fall-F1 (%) | Fall∪Fallen F1 (%) |
|---|---|---|---|
| Baseline (St+Sy, no conditioning) | 26.6 | 45.3 | 55.5 |
| Rule-based Conditioner (bug-fixed) | 30.0 | 45.3 | 55.6 |
| **Hybrid v8 (Learnable + KL-anchor, 3 seeds)** | **34.3 ± 0.6** | 44.6 ± 0.9 | **57.1 ± 0.8** |

Hybrid v8 per-seed: Fallen-F1 = {34.7, 33.5, 34.7}, Fall-F1 = {44.3, 45.8, 43.7}, Fall∪Fallen F1 = {57.9, 57.1, 56.4} (seeds 42/43/44 tương ứng).

### 1b. LaTeX

```latex
\begin{table}[t]
\centering
\caption{Results on OF-ItW test set (OOPS, $n=2804$ segments). Hybrid v8 reported as mean $\pm$ std over 3 seeds (42/43/44).}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
Method & Fallen-F1 (\%) & Fall-F1 (\%) & Fall$\cup$Fallen F1 (\%) \\
\midrule
Baseline (St+Sy)                  & 26.6            & 45.3            & 55.5 \\
Rule-based Conditioner            & 30.0            & 45.3            & 55.6 \\
\textbf{Hybrid v8 (ours)}         & \textbf{34.3 $\pm$ 0.6} & 44.6 $\pm$ 0.9 & \textbf{57.1 $\pm$ 0.8} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 2. Kiểm định thống kê

### 2a. Fall∪Fallen (claim trung tâm của bài) — bootstrap CI + McNemar, video-level

Phương pháp: resample 572 video có hoàn lại × 10,000 lần (giữ nguyên toàn bộ segment của video được chọn để tôn trọng tương quan trong-video), tính lại Fall∪Fallen F1 (Rule vs Hybrid) mỗi lần, CI 95% cho hiệu Hybrid − Rule.

| Seed | Rule F1 | Hybrid F1 | Mean diff | 95% CI | P(Hybrid ≤ Rule) | Kết luận (F1 bootstrap) |
|---|---|---|---|---|---|---|
| 42 | 55.56% | 57.89% | +2.35pp | [+0.90, +3.82] | 0.0017 | **Có ý nghĩa** |
| 43 | 55.56% | 57.13% | +1.57pp | [+0.15, +2.96] | 0.0148 | **Có ý nghĩa** (biên) |
| 44 | 55.56% | 56.41% | +0.86pp | [−0.64, +2.35] | 0.1283 | Không loại trừ được 0 |

McNemar trên union-accuracy (đúng/sai phân loại vào {fall, fallen}), toàn bộ n=2804 segment mỗi seed:

| Seed | both | only_rule | only_hybrid | neither | χ² | p-value |
|---|---|---|---|---|---|---|
| 42 | 1857 | 67 | 147 | 733 | 29.16 | 6.65e-08 |
| 43 | 1849 | 75 | 134 | 746 | 16.10 | 6.02e-05 |
| 44 | 1845 | 79 | 146 | 734 | 19.36 | 1.08e-05 |

**Ghi chú quan trọng — sự khác biệt giữa 2 test cần diễn giải cẩn thận trong Discussion**: McNemar-trên-accuracy cho cả 3 seed đều p<0.0001 (rất mạnh), nhưng bootstrap-CI-trên-F1 cho seed 44 không loại trừ được 0. Đây KHÔNG phải mâu thuẫn — hai test đo hai thứ khác nhau: McNemar đo số lượng segment đổi từ sai→đúng hay đúng→sai (Hybrid thắng rõ ràng ở mọi seed: only_hybrid > only_rule luôn ~2x), còn F1 là tỷ lệ kết hợp precision/recall nên nhạy hơn với cách phân bố lỗi giữa các lớp và với variance của bootstrap ở mẫu có ít segment fallen (318 segment fallen/2804). Kết luận trung thực: **cải thiện Fall∪Fallen là nhất quán và có hướng rõ ràng ở cả 3 seed (Hybrid luôn thắng về accuracy, only_hybrid luôn > only_rule ~2x), nhưng độ lớn của cải thiện tính theo F1 có dao động giữa các seed và không phải lúc nào cũng vượt ngưỡng ý nghĩa thống kê 95% khi đo bằng bootstrap CI.** Nên dùng ngôn ngữ "cải thiện nhất quán, có ý nghĩa thống kê ở 2/3 seed (accuracy: 3/3 seed)" thay vì khẳng định tuyệt đối "luôn có ý nghĩa thống kê ở mọi seed".

### 2b. McNemar per-class (fallen, fall) — Hybrid v8 vs Rule-based, từng seed

| Seed | Class | n | only_rule | only_hybrid | χ² | p-value | Ý nghĩa |
|---|---|---|---|---|---|---|---|
| 42 | fallen (true=fallen) | 318 | 10 | 24 | 4.971 | 0.0258 | Có ý nghĩa (nghiêng Hybrid) |
| 42 | fall (true=fall)     | 636 | 36 | 15 | 7.843 | 0.0051 | Có ý nghĩa (nghiêng **Rule** — đây là cái giá phải trả) |
| 43 | fallen | 318 | 12 | 15 | 0.148 | 0.7003 | Không có ý nghĩa |
| 43 | fall   | 636 | 28 | 29 | 0.000 | 1.0000 | Không có ý nghĩa (không đổi) |
| 44 | fallen | 318 | 13 | 27 | 4.225 | 0.0398 | Có ý nghĩa (nghiêng Hybrid) |
| 44 | fall   | 636 | 50 | 18 | 14.132 | 0.0002 | Có ý nghĩa (nghiêng **Rule** — cái giá phải trả) |

**Diễn giải**: Hybrid v8 cải thiện Fallen-F1 có ý nghĩa thống kê ở 2/3 seed (42, 44), không có ý nghĩa ở seed 43. Đổi lại, Hybrid v8 làm giảm Fall-recall có ý nghĩa thống kê ở 2/3 seed (42, 44) — đây chính là "cái giá" (tradeoff) đã biết và đã báo cáo trước đó, nhưng ở mức nhỏ hơn nhiều so với v6 (Fall-F1 giảm trung bình chỉ 0.7pp so với rule-based, so với mức giảm lớn hơn nhiều ở v6). Điểm mấu chốt: dù có tradeoff cục bộ ở lớp fall, metric tổng hợp lâm sàng quan trọng nhất (Fall∪Fallen — vì cả hai đều cần alarm) vẫn cải thiện nhất quán ở cả 3 seed (mục 2a).

---

## 3. Hình ảnh

- [`figures/confusion_matrix_4class.png`](figures/confusion_matrix_4class.png) — confusion matrix 4 lớp (fall/lie_down/fallen/lying), Baseline vs Hybrid v8 (pooled 3 seed), row-normalized, 300dpi. Định dạng giống Figure 7 paper OmniFall gốc để so sánh trực quan.
- [`figures/comparison_bar_chart.png`](figures/comparison_bar_chart.png) — 3 subplot (Fallen-F1, Fall-F1, Fall∪Fallen F1) so sánh Baseline/Rule-based/Hybrid v8, error bar = std qua 3 seed, kèm scatter điểm từng seed riêng lẻ. 300dpi.

---

## 4. Checkpoint / script tương ứng (để tái tạo hoặc kiểm tra lại)

| Thành phần | Đường dẫn |
|---|---|
| Baseline checkpoint (St+Sy, official repo) | `omnifall-experiments/outputs/2026-09-08/21-55-48/feature_transformer_768_2l/checkpoint-7000/model.safetensors` |
| Baseline cached logits | `data/omnifall/baseline_logits/{stsyn_train,stsyn_val,OOPS_val_stsyn,OOPS_test_stsyn}.pkl` (script: `scripts/cache_baseline_logits_stsyn.py`) |
| Rule-based adjusted logits cache | `data/omnifall/rule_adjusted_logprobs/*.pkl` (script: `scripts/cache_rule_adjusted_logprobs_stsyn.py`, dùng `RuleBasedConditioner` đã sửa 2 bug trong `src/conditioning.py`) |
| Hybrid v8 checkpoints (3 seed) | `checkpoints/learnable_conditioner_v8_stsyn_kl0.3_seed{42,43,44}_best.pt` (train: `scripts/train_learnable_conditioner_v8.py --seed {42,43,44} --lambda_kl 0.3`) |
| Bootstrap CI + McNemar (Fall∪Fallen) | `scripts/bootstrap_fall_union_fallen.py` → `data/omnifall/bootstrap_fall_union_fallen_results.json` |
| McNemar per-class (fallen, fall) | `scripts/mcnemar_per_class_v8.py` → `data/omnifall/mcnemar_per_class_v8_results.json` |
| Hình ảnh cuối | `scripts/make_final_figures.py` → `figures/confusion_matrix_4class.png`, `figures/comparison_bar_chart.png` |
| Báo cáo narrative đầy đủ (lịch sử debug, bug fixes, dataset integration, MCFD negative result) | `KET_QUA_TONG_HOP.md` |

---

## 5. Tóm tắt cho Discussion

- **Đóng góp chính**: Rule-based Conditioner (dựa trên cấu trúc causal của taxonomy: fall→fallen, lie_down→lying) đã sửa 2 bug quan trọng (self-overwrite tại elapsed=0, và ghi đè sai các segment fall liên tiếp) — bản thân nó đã cải thiện Fallen-F1 +3.4pp so với baseline mà không tốn Fall-F1 (45.3%→45.3%, giữ nguyên).
- **Hybrid v8** (Learnable Conditioner + KL-anchor regularization, chọn checkpoint theo Fall∪Fallen thay vì chỉ Fallen-F1) cải thiện thêm: Fallen-F1 +4.3pp so với rule-based (30.0%→34.3%, có ý nghĩa thống kê 2/3 seed), Fall∪Fallen +1.5pp (55.6%→57.1%, có ý nghĩa thống kê ở accuracy cả 3 seed, ở F1-bootstrap 2/3 seed), đổi lại Fall-F1 giảm nhẹ 0.7pp (có ý nghĩa thống kê 2/3 seed).
- **Giới hạn cần nêu trung thực**: (1) OF-ItW test set có rất ít lying/lie_down thật (chỉ 17/11 trong 2804 segment) nên phần lớn signal đến từ lớp fallen/fall — cần nêu rõ đây là giới hạn của test set hiện có, không phải của phương pháp; (2) cải thiện F1 dao động nhẹ giữa seed (không phải tất cả đều vượt ngưỡng 95% CI), nên câu chuyện nên là "cải thiện nhất quán và đáng tin cậy" chứ không phải "luôn luôn có ý nghĩa tuyệt đối ở mọi phép đo".
