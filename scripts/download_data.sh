#!/bin/bash
# Tải dataset OmniFall về ./data/omnifall
#
# LƯU Ý QUAN TRỌNG:
#   - Một số dataset con (CMDFall, UP-Fall, EDF, OCCU, MCFD) yêu cầu permission
#     RIÊNG từ tác giả gốc (xem Appendix A.3 của paper OmniFall) — không tự động
#     tải được, cần liên hệ trực tiếp.
#   - OF-Synthetic và OF-ItW có license CC-BY nên tải trực tiếp được qua HuggingFace.
#   - Cần `huggingface-cli login` trước nếu dataset yêu cầu xác thực.

set -e

mkdir -p data/omnifall
cd data/omnifall

echo "Đang tải OmniFall dataset (OF-Synthetic + OF-ItW + metadata)..."
huggingface-cli download simplexsigil2/omnifall --repo-type dataset --local-dir .

echo ""
echo "⚠️  Nhắc lại: các dataset con staged (CMDFall, UP-Fall, EDF, OCCU, MCFD) có thể"
echo "    cần bạn tự liên hệ tác giả gốc để xin quyền truy cập (xem README của repo)."
echo ""
echo "Sau khi tải xong, kiểm tra cấu trúc thư mục thực tế và chỉnh lại:"
echo "  - src/data.py (hàm _load_annotations, _get_video_path)"
echo "  - src/features.py (cách liệt kê video trong run_extraction)"
