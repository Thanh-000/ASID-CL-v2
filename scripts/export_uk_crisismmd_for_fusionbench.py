"""
Script Chuẩn Bị & Đóng Gói Dataset UK_CrisisMMD_Public Cho fusion_bench & OPCM Continual Merging
==============================================================================================
Mục tiêu:
1. Đọc và làm sạch 3,667 mẫu đa phương thức (Ảnh + Nhãn Khủng hoảng) từ UK_CrisisMMD_Public.
2. Tạo phân chia Stratified Split (Train 70%, Val 10%, Test 20%) bảo toàn phân bố nhãn.
3. Tạo cấu trúc thư mục ImageFolder chuẩn quốc tế cho Hugging Face datasets:
   data/uk_crisismmd_splits/{train, val, test}/{6_classes}
4. Tự động sinh tệp YAML cấu hình cho fusion_bench (Taskpool, Dataset, Templates).
5. Đăng ký Classnames & Prompt Templates chuyên sâu cho CLIP Zero-shot / Fine-tuning.
"""

import json
import os
import sys
import shutil
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# 1. Đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "UK_CrisisMMD_Public"
OUTPUT_DIR = PROJECT_ROOT / "data" / "uk_crisismmd_splits"
JSONL_PATH = DATASET_DIR / "public_dataset.jsonl"
MAPPING_PATH = DATASET_DIR / "image_mapping.csv"

# 6 lớp khủng hoảng chuẩn
CRISIS_CLASSES = [
    "major_accidents",
    "natural_disaster",
    "pandemic_health",
    "political_crisis",
    "social_unrest",
    "strikes",
]

# Prompt Templates tối ưu hóa cho Vision-Language CLIP đối với bài toán cứu nạn khẩn cấp UK
CLIP_TEMPLATES = [
    lambda c: f"a photo of {c}",
    lambda c: f"a news photo showing {c}",
    lambda c: f"a social media image depicting {c} in the United Kingdom",
    lambda c: f"emergency situation showing {c}",
    lambda c: f"a photo capturing {c}",
]

# Tên mô tả chi tiết nhãn để CLIP Text Encoder sinh vector ngữ nghĩa chuẩn xác
CLIP_CLASS_DESCRIPTIONS = {
    "major_accidents": "a major accident, transportation crash, or severe structural fire",
    "natural_disaster": "a natural disaster, severe river flooding, storm damage, or weather emergency",
    "pandemic_health": "a pandemic health emergency, coronavirus covid-19 medical crisis, or healthcare vaccination",
    "political_crisis": "a political crisis, government protest, or parliamentary controversy",
    "social_unrest": "social unrest, civil street protest, or public riot",
    "strikes": "workers on strike, industrial picket line, trade union rally, or strike protest banner",
}

def load_clean_multimodal_records():
    print(f"📖 Đang đọc dữ liệu từ: {JSONL_PATH}")
    records = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            img_rel = item.get("image_path")
            ctype = item.get("predicted_crisis_type")
            if img_rel and ctype and ctype in CRISIS_CLASSES:
                full_img_path = DATASET_DIR / img_rel
                if full_img_path.exists():
                    records.append({
                        "id": idx,
                        "text": item.get("text", "").strip(),
                        "image_path": str(full_img_path),
                        "crisis_type": ctype,
                        "informativeness": item.get("predicted_informativeness", "uninformative"),
                        "platform": item.get("platform", "unknown"),
                        "source_query": item.get("source_query", ""),
                    })
    df = pd.DataFrame(records)
    print(f"✅ Đã tải và xác thực thành công {len(df)} mẫu ảnh hợp lệ 100%!")
    return df

def create_stratified_splits(df, seed=42):
    print("\n⚖ Đang chia tập Train (70%) / Val (10%) / Test (20%) theo phân tầng lớp...")
    rng = np.random.default_rng(seed)
    
    df["split"] = "train"
    train_indices = []
    val_indices = []
    test_indices = []
    
    for ctype in CRISIS_CLASSES:
        c_idxs = df[df["crisis_type"] == ctype].index.values.copy()
        rng.shuffle(c_idxs)
        n = len(c_idxs)
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        
        train_indices.extend(c_idxs[:n_train])
        val_indices.extend(c_idxs[n_train:n_train+n_val])
        test_indices.extend(c_idxs[n_train+n_val:])
        
    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "val"
    df.loc[test_indices, "split"] = "test"
    
    print("\n📊 Phân bố số lượng mẫu theo từng Split:")
    summary_table = []
    for ctype in CRISIS_CLASSES:
        row = {"Crisis Type": ctype}
        for s in ["train", "val", "test"]:
            cnt = len(df[(df["crisis_type"] == ctype) & (df["split"] == s)])
            row[s.capitalize()] = cnt
        row["Total"] = len(df[df["crisis_type"] == ctype])
        summary_table.append(row)
        
    sum_df = pd.DataFrame(summary_table)
    print(sum_df.to_string(index=False))
    return df

def export_imagefolder_structure(df):
    print(f"\n📂 Đang xuất cấu trúc ImageFolder chuẩn vào: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ["train", "val", "test"]:
        for ctype in CRISIS_CLASSES:
            target_dir = OUTPUT_DIR / split / ctype
            target_dir.mkdir(parents=True, exist_ok=True)
            
    # Sao chép/Liên kết ảnh vào thư mục split
    exported_count = 0
    for _, row in df.iterrows():
        src_path = Path(row["image_path"])
        dst_path = OUTPUT_DIR / row["split"] / row["crisis_type"] / src_path.name
        try:
            # Thử tạo hardlink trên NTFS để 0 tốn thêm dung lượng đĩa
            os.link(src_path, dst_path)
        except Exception:
            shutil.copy2(src_path, dst_path)
        exported_count += 1
        
    # Lưu metadata CSV cho từng split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        split_csv = OUTPUT_DIR / f"{split}_metadata.csv"
        split_df.to_csv(split_csv, index=False, encoding="utf-8")
        
    print(f"✅ Hoàn tất xuất {exported_count} ảnh vào {OUTPUT_DIR}!")

def generate_fusion_bench_configs():
    print("\n⚙ Đang sinh các tệp cấu hình Hydra cho fusion_bench & OPCM...")
    
    # 1. Sinh tệp cấu hình dataset test
    cfg_dataset_dir = PROJECT_ROOT / "external" / "opcm_author_repo" / "config" / "dataset" / "image_classification" / "test"
    cfg_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    uk_dataset_yaml = cfg_dataset_dir / "uk_crisis.yaml"
    uk_dataset_yaml_content = f"""uk_crisis:
  _target_: datasets.load_dataset
  path: imagefolder
  data_dir: "{OUTPUT_DIR.as_posix()}"
  split: test
"""
    uk_dataset_yaml.write_text(uk_dataset_yaml_content, encoding="utf-8")
    print(f" - Đã tạo: {uk_dataset_yaml}")
    
    # 2. Sinh cấu hình CLIP Taskpool
    cfg_taskpool_dir = PROJECT_ROOT / "external" / "opcm_author_repo" / "config" / "taskpool" / "CLIPVisionModelTaskPool"
    cfg_taskpool_dir.mkdir(parents=True, exist_ok=True)
    
    uk_taskpool_yaml = cfg_taskpool_dir / "clip-vit-classification_UK_CRISIS.yaml"
    uk_taskpool_yaml_content = """defaults:
  - CLIPVisionModelTaskPool@: _template
  - /dataset/image_classification/test@test_datasets:
      - uk_crisis

base_model: openai/clip-vit-base-patch16
"""
    uk_taskpool_yaml.write_text(uk_taskpool_yaml_content, encoding="utf-8")
    print(f" - Đã tạo: {uk_taskpool_yaml}")
    
    # 3. Tạo mô đun định nghĩa Classnames & Prompt Templates trong fusion_bench
    task_tpl_dir = PROJECT_ROOT / "external" / "opcm_author_repo" / "fusion_bench" / "tasks" / "clip_classification"
    task_tpl_file = task_tpl_dir / "uk_crisis.py"
    
    classnames_code = "classnames = [\n" + ",\n".join([f'    "{CLIP_CLASS_DESCRIPTIONS[c]}"' for c in CRISIS_CLASSES]) + "\n]\n\n"
    templates_code = "templates = [\n" + ",\n".join([
        '    lambda c: f"a photo of {c}",',
        '    lambda c: f"a news photo showing {c}",',
        '    lambda c: f"a social media image depicting {c} in the United Kingdom",',
        '    lambda c: f"emergency situation showing {c}",',
        '    lambda c: f"a photo capturing {c}",',
    ]) + "\n]\n"
    
    task_tpl_file.write_text(classnames_code + templates_code, encoding="utf-8")
    print(f" - Đã tạo mô đun ngữ nghĩa CLIP: {task_tpl_file}")
    
    # 4. Đăng ký uk_crisis vào CLIPTemplateFactory
    init_file = task_tpl_dir / "__init__.py"
    if init_file.exists():
        init_code = init_file.read_text(encoding="utf-8")
        if '"uk_crisis": ".uk_crisis"' not in init_code:
            target_str = '"mango-leaf-disease": ".mango_leaf_disease",'
            replacement = '        "mango-leaf-disease": ".mango_leaf_disease",\n        "uk_crisis": ".uk_crisis",'
            if target_str in init_code:
                init_code = init_code.replace(target_str, replacement)
                init_file.write_text(init_code, encoding="utf-8")
                print(" - Đã đăng ký 'uk_crisis' vào CLIPTemplateFactory!")

if __name__ == "__main__":
    print("="*80)
    print("🚀 BẮT ĐẦU QUY TRÌNH CHUẨN BỊ UK_CRISISMMD CHO CONTINUAL MODEL MERGING")
    print("="*80)
    df = load_clean_multimodal_records()
    df_splits = create_stratified_splits(df)
    export_imagefolder_structure(df_splits)
    generate_fusion_bench_configs()
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH 100%! DATASET ĐÃ SẴN SÀNG CHO FUSION_BENCH & OPCM!")
    print("="*80)
