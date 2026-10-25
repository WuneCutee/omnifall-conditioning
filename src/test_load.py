"""
Script kiểm tra nhanh: config nào của OmniFall thực sự load được, và cấu trúc
1 example thật trông như thế nào — chạy trước khi viết lại src/data.py.

Usage:
    python scripts/test_load.py
"""
from datasets import load_dataset

# Thử lần lượt các config, từ "an toàn nhất" (chắc chắn có trong builder.py)
# đến config mong muốn theo CONFIGS.md (có thể chưa được implement)
CONFIGS_TO_TRY = [
    "of-sta-cs",          # OF-Staged only, chắc chắn có trong builder.py
    "of-syn",             # OF-Synthetic only, chắc chắn có
    "of-itw",             # OF-ItW only, chắc chắn có (nhưng cần video riêng)
    "of-sta-itw-cs",      # Cross-domain tên cũ, có trong builder.py đã đọc
    "of-sta-syn-to-all-cs",  # Tên mong muốn theo CONFIGS.md -- có thể lỗi
    "of-syn-itw",         # Cross-domain tên cũ khác
]

for config_name in CONFIGS_TO_TRY:
    print(f"\n{'='*60}")
    print(f"Thử load config: '{config_name}'")
    print("=" * 60)
    try:
        ds = load_dataset("simplexsigil2/omnifall", config_name, trust_remote_code=True)
        print(f"✅ THÀNH CÔNG. Các split: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            print(f"  - {split_name}: {len(split_data)} examples")
        # In thử 1 example đầu tiên của split đầu tiên
        first_split = list(ds.keys())[0]
        print(f"\n  Ví dụ 1 dòng từ split '{first_split}':")
        print(f"  {ds[first_split][0]}")
    except Exception as e:
        print(f"❌ LỖI: {type(e).__name__}: {e}")