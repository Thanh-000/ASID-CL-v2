import json
import os

notebook = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {
            'gpuType': 'T4',
            'provenance': []
        },
        'language_info': {
            'name': 'python'
        }
    },
    'cells': []
}

def add_md(source):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in source.split('\n')]
    })

def add_code(source):
    notebook['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in source.split('\n')]
    })

# Cell 1: Intro
add_md("""# 🚀 TÁI LẬP THỰC NGHIỆM ĐẦY ĐỦ OPCM TRÊN GOOGLE COLAB
### Paper: *Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable Continual Model Merging*
**Tác giả:** Anke Tang, Enneng Yang, Li Shen, Yong Luo, Han Hu, Bo Du, Dacheng Tao (arXiv:2501.09522v1)

**Mục tiêu Notebook:** Tái hiện hoàn chỉnh chu trình thực nghiệm Continual Model Merging trên Google Colab GPU (T4 / V100 / A100), bao gồm:
1. Thiết lập môi trường và clone repository chính thức `tanganke/opcm`.
2. Áp dụng bản vá phẫu thuật lỗi in-place PyTorch (`diagonal().zero_()`).
3. Chạy 3 phương pháp đối chứng: **OPCM (Đề xuất)**, **Task Arithmetic**, và **Simple Weight Averaging (SWA)** trên backbone ViT-B/32 (Base 8 Tasks).
4. Phân tích kết quả, tính toán **Final Accuracy (ACC)** và **Backward Transfer (BWT)**, đối chiếu trực tiếp với **Table 1** trong bài báo gốc.
5. Trực quan hóa biểu đồ học liên tục (Learning Curves) và ma trận suy giảm tri thức (Accuracy Heatmap).""")

# Cell 2: Check GPU
add_code("""# [1] KIỂM TRA PHẦN CỨNG GPU TRÊN GOOGLE COLAB
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("CẢNH BÁO: Bạn đang chạy trên CPU! Hãy vào Runtime -> Change runtime type -> Chọn T4 GPU hoặc A100 GPU.")""")

# Cell 3: Install dependencies
add_code("""# [2] CÀI ĐẶT CÁC THƯ VIỆN PHỤ THUỘC (DEPENDENCIES)
!pip install --quiet lightning peft transformers datasets hydra-core omegaconf matplotlib seaborn pandas tqdm rich scipy h5py
print("Đã cài đặt thành công toàn bộ thư viện cần thiết!")""")

# Cell 4: Clone repo and install
add_code("""# [3] CLONE REPOSITORY CHÍNH THỨC CỦA TÁC GIẢ & CÀI ĐẶT GÓI FUSION_BENCH
import os

if not os.path.exists("opcm"):
    !git clone https://github.com/tanganke/opcm.git
%cd opcm
!pip install --quiet -e .
print("Đã cài đặt fusion_bench thành công!")""")

# Cell 5: Bugfix
add_code("""# [4] ÁP DỤNG BẢN VÁ PHẪU THUẬT LỖI IN-PLACE PYTORCH (THEOREM 5.1 COMPLIANCE)
# File: fusion_bench/method/opcm/opcm.py dòng 169
# Lỗi gốc của tác giả: projected_task_tv.diag().fill_(0) -> Chỉ sửa tensor 1D tạm thời!
# Bản vá chuẩn xác: projected_task_tv.diagonal().zero_() -> Làm 0 đường chéo ma trận 2D gốc!

opcm_file = "fusion_bench/method/opcm/opcm.py"
with open(opcm_file, "r", encoding="utf-8") as f:
    code = f.read()

if "projected_task_tv.diag().fill_(0)" in code:
    code = code.replace(
        "projected_task_tv.diag().fill_(0)",
        "projected_task_tv.diagonal().zero_()"
    )
    with open(opcm_file, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ THÀNH CÔNG: Đã vá lỗi PyTorch in-place bằng .diagonal().zero_()!")
elif "projected_task_tv.diagonal().zero_()" in code:
    print("ℹ File đã được vá từ trước (.diagonal().zero_()).")
else:
    print("⚠ Cảnh báo: Không tìm thấy dòng mã đích để vá. Vui lòng kiểm tra lại opcm.py.")

# Unit check nhanh kiểm chứng tính trực giao
test_mat = torch.randn(8, 8)
diag_copy = test_mat.diag()
diag_copy.fill_(0)
buggy_zero = (torch.diagonal(test_mat).abs().sum().item() == 0)

test_mat_fixed = torch.randn(8, 8)
test_mat_fixed.diagonal().zero_()
fixed_zero = (torch.diagonal(test_mat_fixed).abs().sum().item() == 0)

print(f"Kiểm chứng hàm .diag().fill_(0) gốc: Đường chéo bằng 0? -> {buggy_zero} (LỖI!)")
print(f"Kiểm chứng bản vá .diagonal().zero_(): Đường chéo bằng 0? -> {fixed_zero} (CHUẨN XÁC!)")""")

# Cell 5.5: Dataset Ingestion and Pre-caching
add_md("""## 📦 QUẢN LÝ & NẠP DATASET (DATASET INGESTION & PRE-CACHING)
Toàn bộ dữ liệu đánh giá 8 tác vụ được lưu trữ công khai trên Hugging Face Hub dưới namespace `tanganke/<dataset_name>`:
1. `tanganke/sun397`: Phân loại cảnh quan (21,750 ảnh test)
2. `tanganke/stanford-cars`: Phân loại xe hơi (8,041 ảnh test)
3. `tanganke/resisc45`: Ảnh vệ tinh độ phân giải cao (6,300 ảnh test)
4. `tanganke/eurosat`: Ảnh vệ tinh quang học Sentinel-2 (5,400 ảnh test)
5. `tanganke/svhn`: Chữ số nhà ngoài đời thực (26,032 ảnh test)
6. `tanganke/gtsrb`: Biển báo giao thông Đức (12,630 ảnh test)
7. `tanganke/mnist`: Chữ số viết tay (10,000 ảnh test)
8. `tanganke/dtd`: Họa tiết tự nhiên Describable Textures (1,880 ảnh test)

Khối mã dưới đây tải trước toàn bộ 8 tập test vào cache cục bộ (`~/.cache/huggingface/datasets/`) giúp quá trình chạy thực nghiệm bên dưới không bao giờ bị nghẽn mạng hoặc timeout.""")

add_code("""# [4.5] TẢI TRƯỚC VÀ KIỂM TRA TOÀN BỘ 8 TEST DATASETS
from datasets import load_dataset
from tqdm.auto import tqdm

test_datasets_list = [
    "sun397", "stanford-cars", "resisc45", "eurosat",
    "svhn", "gtsrb", "mnist", "dtd"
]

print("=== TIẾN HÀNH TẢI TRƯỚC VÀ CACHE 8 TEST DATASETS ===")
dataset_stats = {}

for name in tqdm(test_datasets_list, desc="Pre-caching Datasets"):
    repo_path = f"tanganke/{name}"
    ds = load_dataset(repo_path, split="test")
    dataset_stats[name] = len(ds)
    print(f" -> [OK] {repo_path:30s} | Test split size: {len(ds):,} images")

print("\\n✅ TOÀN BỘ 8 DATASETS ĐÃ ĐƯỢC CACHE THÀNH CÔNG VÀO BỘ NHỚ LOCAL!")
print(f"Tổng số ảnh test sẽ được đánh giá qua các bước: {sum(dataset_stats.values()):,} ảnh.")""")

# Cell 6: Experiment intro
add_md("""## 🧪 THIẾT LẬP THÍ NGHIỆM ĐỐI CHỨNG (ViT-B/32 trên Base 8 Tasks)
Chúng ta sẽ chạy 3 thí nghiệm đối chứng với cùng cấu hình chuẩn:
- **Kiến trúc mô hình:** CLIP-ViT-B/32 (Base model: `openai/clip-vit-base-patch32`)
- **Tập tác vụ (8 Tasks):** SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD.
- **Phương pháp 1 (Đề xuất):** OPCM (Sequential SVD Projection với α = 0.5)
- **Phương pháp 2 (Baseline 1):** Continual Task Arithmetic (với scaling factor λ = 0.3)
- **Phương pháp 3 (Baseline 2):** Simple Weight Averaging (SWA)""")

# Cell 7: Run OPCM
add_code("""# [5] CHẠY THỰC NGHIỆM 1: OPCM (ĐÃ VÁ LỖI, ALPHA = 0.5)
!fusion_bench \\
    fabric.loggers.root_dir=outputs/reproduce \\
    fabric.loggers.name=vit-b-32-TA8-opcm \\
    fabric.loggers.version=0 \\
    method=opcm/sequential_projection \\
    method.alpha=0.5 \\
    method.seed=42 \\
    method.shuffle_order=false \\
    method.save_on_every_step=false \\
    method.evaluate_on_every_step=true \\
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \\
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

print("✅ Đã hoàn thành thực nghiệm OPCM!")""")

# Cell 8: Run TA
add_code("""# [6] CHẠY THỰC NGHIỆM 2: CONTINUAL TASK ARITHMETIC (LAMBDA = 0.3)
!fusion_bench \\
    fabric.loggers.root_dir=outputs/reproduce \\
    fabric.loggers.name=vit-b-32-TA8-task-arithmetic \\
    fabric.loggers.version=0 \\
    method=opcm/task_arithmetic \\
    method.scaling_factor=0.3 \\
    method.seed=42 \\
    method.shuffle_order=false \\
    method.save_on_every_step=false \\
    method.evaluate_on_every_step=true \\
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \\
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

print("✅ Đã hoàn thành thực nghiệm Continual Task Arithmetic!")""")

# Cell 9: Run SWA
add_code("""# [7] CHẠY THỰC NGHIỆM 3: SIMPLE WEIGHT AVERAGING (SWA)
!fusion_bench \\
    fabric.loggers.root_dir=outputs/reproduce \\
    fabric.loggers.name=vit-b-32-TA8-weight-average \\
    fabric.loggers.version=0 \\
    method=opcm/weight_average \\
    method.seed=42 \\
    method.shuffle_order=false \\
    method.save_on_every_step=false \\
    method.evaluate_on_every_step=true \\
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \\
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

print("✅ Đã hoàn thành thực nghiệm Simple Weight Averaging!")""")

# Cell 10: Analyze results and generate Table 1 comparison
add_code("""# [8] PHÂN TÍCH KẾT QUẢ ĐỊNH LƯỢNG & ĐỐI CHIẾU TRỰC TIẾP VỚI TABLE 1
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def extract_metrics(run_dir, num_tasks=8):
    \"\"\"
    Đọc tất cả các file report_*.json từ bước 0 đến num_tasks-1,
    trích xuất ma trận R[t, i] (độ chính xác task i tại bước t).
    \"\"\"
    R = np.zeros((num_tasks, num_tasks))
    task_names = []
    
    for t in range(num_tasks):
        report_file = Path(run_dir) / f"report_{t}.json"
        if not report_file.exists():
            print(f"Cảnh báo: Không tìm thấy {report_file}")
            continue
        with open(report_file, "r") as f:
            data = json.load(f)
            
        # Trích xuất accuracy của các task đã học đến bước t
        acc_dict = {}
        for k, v in data.items():
            if isinstance(v, dict) and "acc" in v:
                acc_dict[k] = v["acc"]
            elif isinstance(v, (int, float)) and "acc" in k.lower():
                acc_dict[k] = v
                
        if t == 0:
            task_names = list(acc_dict.keys())
        for i, name in enumerate(task_names[:t+1]):
            if name in acc_dict:
                R[t, i] = acc_dict[name] * 100.0 if acc_dict[name] <= 1.0 else acc_dict[name]
                
    # Tính Final Average Accuracy tại bước cuối cùng T = num_tasks - 1
    final_acc = np.mean(R[num_tasks - 1, :])
    
    # Tính Backward Transfer (BWT)
    # BWT = (1 / (T - 1)) * sum_{i=0}^{T-2} (R[T-1, i] - R[i, i])
    bwt_diffs = [R[num_tasks - 1, i] - R[i, i] for i in range(num_tasks - 1)]
    bwt = np.mean(bwt_diffs) if len(bwt_diffs) > 0 else 0.0
    
    return final_acc, bwt, R, task_names

# Phân tích cả 3 phương pháp
methods = {
    "OPCM (Ours - Patched)": "outputs/reproduce/vit-b-32-TA8-opcm/version_0",
    "Continual Task Arithmetic": "outputs/reproduce/vit-b-32-TA8-task-arithmetic/version_0",
    "Simple Weight Averaging (SWA)": "outputs/reproduce/vit-b-32-TA8-weight-average/version_0"
}

results = []
matrices = {}

for name, path in methods.items():
    if os.path.exists(path):
        f_acc, bwt, R_mat, t_names = extract_metrics(path, num_tasks=8)
        results.append({"Method": name, "Re-produced Final ACC (%)": f_acc, "Re-produced BWT (%)": bwt})
        matrices[name] = (R_mat, t_names)
    else:
        print(f"Thư mục {path} chưa có kết quả.")

df_results = pd.DataFrame(results)

# Bảng đối chiếu với Table 1 trong bài báo gốc
paper_table1 = pd.DataFrame([
    {"Method": "OPCM (Ours - Patched)", "Paper Final ACC (%)": 75.5, "Paper BWT (%)": -6.3},
    {"Method": "Continual Task Arithmetic", "Paper Final ACC (%)": 67.5, "Paper BWT (%)": -9.6},
    {"Method": "Simple Weight Averaging (SWA)", "Paper Final ACC (%)": 66.3, "Paper BWT (%)": -11.5}
])

if len(df_results) > 0:
    comparison_df = pd.merge(df_results, paper_table1, on="Method")
    print("\\n=================== BẢNG ĐỐI CHIẾU KẾT QUẢ VỚI TABLE 1 PAPER ===================")
    display(comparison_df)
else:
    print("Chưa có kết quả để hiển thị.")""")

# Cell 11: Plots
add_code("""# [9] TRỰC QUAN HÓA: LEARNING CURVES & ACCURACY HEATMAP (GIỐNG FIGURE 5 & 22 TRONG PAPER)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Đồ thị 1: Average Accuracy theo số lượng tác vụ (t = 1 đến 8)
ax1 = axes[0]
for name, (R_mat, _) in matrices.items():
    # Tính accuracy trung bình tại mỗi bước t
    step_means = [np.mean(R_mat[t, :t+1]) for t in range(8)]
    ax1.plot(range(1, 9), step_means, marker="o", linewidth=2.5, label=name)

ax1.set_title("Average Accuracy Across Sequential Merging Steps", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Merged Tasks (t)", fontsize=11)
ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
ax1.set_xticks(range(1, 9))
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.6)

# Đồ thị 2: Heatmap ma trận độ chính xác R[t, i] của OPCM
ax2 = axes[1]
if "OPCM (Ours - Patched)" in matrices:
    R_opcm, t_names = matrices["OPCM (Ours - Patched)"]
    # Tạo mặt nạ tam giác trên
    mask = np.triu(np.ones_like(R_opcm, dtype=bool), k=1)
    sns.heatmap(
        R_opcm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        mask=mask,
        ax=ax2,
        cbar_kws={"label": "Accuracy (%)"},
        xticklabels=[f"T{i+1}" for i in range(8)],
        yticklabels=[f"Step {t+1}" for t in range(8)]
    )
    ax2.set_title("OPCM Task Accuracy Matrix R[t, i] (Continual Retention)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Evaluated Task Index (i)", fontsize=11)
    ax2.set_ylabel("Current Merged Step (t)", fontsize=11)

plt.tight_layout()
plt.savefig("opcm_reproduction_results.png", dpi=300)
plt.show()
print("Đã lưu biểu đồ vào opcm_reproduction_results.png!")""")

# Cell 12: Export to Drive
add_code("""# [10] XUẤT TOÀN BỘ KẾT QUẢ VÀ BÁO CÁO VÀO GOOGLE DRIVE
from google.colab import drive
drive.mount("/content/drive")

# Nén thư mục kết quả và hình ảnh
!zip -r opcm_reproduction_bundle.zip outputs/ opcm_reproduction_results.png

# Copy vào Google Drive
save_drive_path = "/content/drive/MyDrive/OPCM_Reproduction/"
os.makedirs(save_drive_path, exist_ok=True)
!cp opcm_reproduction_bundle.zip "{save_drive_path}"
print(f"✅ ĐÃ XUẤT THÀNH CÔNG: Toàn bộ báo cáo đã được lưu vào Google Drive tại: {save_drive_path}")""")

# Write file
output_path = 'OPCM_Full_Reproduction_Google_Colab.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Created notebook successfully: {output_path}")
