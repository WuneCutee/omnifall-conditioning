#!/bin/bash
# Bước 2: Pilot nhanh — áp rule-based conditioning lên baseline đã train,
# KHÔNG cần train lại. Đây là bước "kiểm tra hypothesis rẻ tiền" trước khi
# đầu tư vào bản learnable (Bước 3).
set -e

if [ ! -f checkpoints/baseline_best.pt ]; then
    echo "Chưa có checkpoint baseline. Chạy scripts/run_baseline.sh trước."
    exit 1
fi

echo "=== So sánh: Baseline thuần (không conditioning) ==="
python -m src.evaluate --config configs/default.yaml \
    --checkpoint checkpoints/baseline_best.pt --model-type baseline

echo ""
echo "=== So sánh: Baseline + Rule-based Conditioning ==="
python -m src.evaluate --config configs/default.yaml \
    --checkpoint checkpoints/baseline_best.pt --model-type rule_based_pilot

echo ""
echo "Nếu Fallen-F1 của bản conditioning CAO HƠN rõ rệt so với baseline thuần,"
echo "hypothesis có tín hiệu tốt -> tiến hành Bước 3 (train src.train --model conditioned)."
echo "Nếu KHÔNG cải thiện hoặc cải thiện không đáng kể, cần xem lại hypothesis"
echo "trước khi đầu tư thêm thời gian vào bản learnable."
