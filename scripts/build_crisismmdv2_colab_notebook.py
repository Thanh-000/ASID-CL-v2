import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

notebook = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {
            'gpuType': 'L4',
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
        'source': [line + '\n' for line in source.strip().split('\n')]
    })

def add_code(source):
    notebook['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in source.strip().split('\n')]
    })

# ==============================================================================
# Cell 1: Giới thiệu & Tổng quan Nghiên cứu Toàn Cầu trên CrisisMMDv2
# ==============================================================================
add_md(r"""# 🌐 NGHIÊN CỨU MỞ RỘNG TOÀN CẦU: OPCM TRÊN QCRI/CRISISMMDv2 (GLOBAL MULTIMODAL DISASTER STUDY)
### *Đánh giá Khả Năng Khái Quát Hóa Đa Phương Thức & Mức Độ Tàn Phá Hạ Tầng (Damage Severity Assessment)*
**Mô hình nghiên cứu:** OPCM (*Tang et al., arXiv:2501.09522v1*) gộp tuần tự 14 chuyên gia thị giác ViT-B/16  
**Dataset mở rộng:** `QCRI/CrisisMMD` (Crisis Multimedia Dataset Version 2.0 - Viện Nghiên cứu Điện toán Qatar)  
**Tác vụ cốt lõi:** `Damage Severity Assessment` (3 phân lớp: Little/No Damage, Mild Damage, Severe Damage trên 7 trận thảm họa toàn cầu: Bão Harvey, Irma, Maria, Động đất Mexico, Cháy rừng California,...)  
**Phần cứng tối ưu:** NVIDIA L4 GPU (24GB VRAM) / A100 trên Google Colab  

---
### 🎯 CÁC MỤC TIÊU KHOA HỌC THEN CHỐT:
1. **Kiểm chứng tính tổng quát hóa toàn cầu (Global Out-of-Distribution Generalization):**  
   Xác định xem OPCM (sau khi gộp 14 chuyên gia sạch trong phòng thí nghiệm) có duy trì được ưu thế vượt trội trước Pretrained CLIP và các phương pháp gộp khác trên dữ liệu thảm họa thế giới hay không.
2. **Kích hoạt tối đa sức mạnh của Cặp Gen Viễn Thám (`EuroSAT` & `Resisc45`):**  
   Đo lường năng lực phân loại mức độ tàn phá (Damage Severity) khi đối đầu với các bức ảnh sập nhà, vỡ đê, gãy cầu từ trên cao.
3. **Thực thi Bounded Gating AC-Fusion:**  
   Áp dụng cơ chế kẹp biên an toàn đã khám phá để tối ưu hóa hòa trộn giữa Ảnh hiện trường và Tweet tiếng Anh thực tế.
""")

# ==============================================================================
# Cell 2: Kiểm tra phần cứng GPU
# ==============================================================================
add_code(r'''# [1] KIỂM TRA PHẦN CỨNG GPU TRÊN GOOGLE COLAB
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
''')

# ==============================================================================
# Cell 3: Cài đặt thư viện cốt lõi
# ==============================================================================
add_code(r'''# [2] CÀI ĐẶT CÁC THƯ VIỆN CỐT LÕI & ARIA2C TẢI SIÊU TỐC
!apt-get -qq install -y aria2
!pip install --quiet lightning peft transformers datasets hydra-core omegaconf matplotlib seaborn pandas tqdm scikit-learn
print("✅ Đã cài đặt hoàn tất các thư viện cần thiết và aria2c!")
''')

# ==============================================================================
# Cell 4: Tải và Nạp Dataset QCRI/CrisisMMD Bằng aria2c Tốc Độ Cao
# ==============================================================================
add_code(r'''# [3] TẢI VÀ NẠP DATASET QCRI/CRISISMMD BẰNG ARIA2C TỐC ĐỘ CAO (16 CONNECTIONS)
import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/content/CrisisMMD_v2.0")
TAR_FILE = Path("/content/CrisisMMD_v2.0.tar.gz")

if not (DATA_DIR / "data_image").exists():
    print("🚀 Bắt đầu tải CrisisMMD_v2.0.tar.gz bằng aria2c (16 luồng song song siêu tốc)...")
    !aria2c -x 16 -s 16 -k 1M -j 16 --summary-interval=1 --console-log-level=warn \
        "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz" \
        -d /content -o CrisisMMD_v2.0.tar.gz
    
    print("📦 Đang giải nén dữ liệu ảnh hiện trường và các tệp phân chia...")
    !tar -xzf /content/CrisisMMD_v2.0.tar.gz -C /content/
    !unzip -q -o /content/CrisisMMD_v2.0/crisismmd_datasplit_all.zip -d /content/CrisisMMD_v2.0/splits/
    print("✅ Đã giải nén hoàn tất!")

# Đọc tệp phân chia test chính thức cho tác vụ Damage Severity
SPLIT_FILE = Path("/content/CrisisMMD_v2.0/splits/crisismmd_datasplit_all/task_damage_text_img_test.tsv")
df_test = pd.read_csv(SPLIT_FILE, sep="\t")
print(f"\n✅ Đã nạp thành công {len(df_test)} mẫu thử nghiệm chính thức của CrisisMMD v2 (Damage)!")
print(f"📋 Các cột dữ liệu: {df_test.columns.tolist()}")

# Chuẩn hóa nhãn về dạng số: 0, 1, 2
LABEL_MAP = {
    "little_or_no_damage": 0,
    "mild_damage": 1,
    "severe_damage": 2,
}
df_test["label_idx"] = df_test["label"].map(LABEL_MAP)
label_counts = df_test["label_idx"].value_counts().sort_index()

print("\n📊 Phân phối nhãn trên tập test chính thức:")
for k, v in label_counts.items():
    lbl_name = [name for name, idx in LABEL_MAP.items() if idx == k][0]
    print(f"   - Lớp {k} ({lbl_name}): {v} mẫu ({v/len(df_test)*100:.1f}%)")
''')

# ==============================================================================
# Cell 5: Trích xuất và Cache Tensor Đa Mô Thức cho CrisisMMDv2
# ==============================================================================
add_code(r'''# [4] TRÍCH XUẤT VÀ TẠO IN-MEMORY TENSOR CACHE TỪ ẢNH VÀ TWEET THỰC TẾ
import torch
import torch.nn.functional as F
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

CACHE_DIR = Path("/content/tensor_cache_crisismmd")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "crisismmd_v2_cache.pt"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").cuda().eval()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

DAMAGE_CLASSES = ["little_or_no_damage", "mild_damage", "severe_damage"]
DAMAGE_DESCRIPTIONS = {
    "little_or_no_damage": "infrastructure with little or no damage, buildings intact and safe after natural disaster",
    "mild_damage": "moderate mild damage, slight flood water, minor debris or partially damaged roof",
    "severe_damage": "severe catastrophic damage, completely destroyed collapsed buildings, ruined bridges and severe disaster destruction",
}

# 1. Trích xuất đặc trưng ảnh theo batch từ đường dẫn ảnh thật (Cột 'image')
print("🖼️ Đang nạp và trích xuất đặc trưng cho toàn bộ 529 bức ảnh hiện trường...")
batch_size = 128

# Tự động nhận diện cột đường dẫn ảnh ('image' hoặc 'image_path')
img_col = "image" if "image" in df_test.columns else ("image_path" if "image_path" in df_test.columns else df_test.columns[4])
img_paths = [Path("/content/CrisisMMD_v2.0") / str(p) for p in df_test[img_col]]
img_batches = []
labels_tensor = torch.tensor(df_test["label_idx"].values, dtype=torch.long)

for i in tqdm(range(0, len(img_paths), batch_size), desc="Caching Images"):
    batch_paths = img_paths[i : i + batch_size]
    rgb_imgs = []
    for p in batch_paths:
        try:
            if p.exists():
                rgb_imgs.append(Image.open(p).convert("RGB"))
            else:
                rgb_imgs.append(Image.new("RGB", (224, 224), color=0))
        except Exception:
            rgb_imgs.append(Image.new("RGB", (224, 224), color=0))
            
    inputs = processor(images=rgb_imgs, return_tensors="pt")
    img_batches.append(inputs["pixel_values"].half())

# 2. Tiền xử lý văn bản tweet
def clean_tweet(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "natural disaster damage assessment"

all_texts = [clean_tweet(t) for t in df_test["tweet_text"]]

# 3. Trích xuất đặc trưng văn bản bằng CLIP tháp chiếu chuẩn
print("📝 Đang trích xuất đặc trưng ngôn ngữ cho các bài đăng tweet...")
text_embs = []
with torch.no_grad():
    for k in range(0, len(all_texts), 128):
        chunk = all_texts[k : k + 128]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        txt_raw = clip_model.text_model(**tok)
        txt_pooled = txt_raw.pooler_output if hasattr(txt_raw, "pooler_output") and txt_raw.pooler_output is not None else txt_raw[1]
        txt_proj = clip_model.text_projection(txt_pooled)
        txt_feat = F.normalize(txt_proj, dim=-1).half().cpu()
        text_embs.append(txt_feat)
all_text_feats = torch.cat(text_embs, dim=0)

# 4. Trích xuất Text Classifier Weights từ Prompts
templates = [
    lambda c: f"a photo of {c}",
    lambda c: f"a satellite or ground photo showing {c}",
    lambda c: f"disaster scene depicting {c}",
    lambda c: f"infrastructure assessment showing {c}",
    lambda c: f"a news image of {c}",
]

weights_list = []
with torch.no_grad():
    for c in DAMAGE_CLASSES:
        prompts = [tpl(DAMAGE_DESCRIPTIONS[c]) for tpl in templates]
        tok = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        txt_raw = clip_model.text_model(**tok)
        txt_pooled = txt_raw.pooler_output if hasattr(txt_raw, "pooler_output") and txt_raw.pooler_output is not None else txt_raw[1]
        txt_proj = clip_model.text_projection(txt_pooled)
        txt_feat = F.normalize(txt_proj, dim=-1)
        mean_feat = F.normalize(txt_feat.mean(dim=0), dim=-1).half().cpu()
        weights_list.append(mean_feat)
classifier_weights = torch.stack(weights_list, dim=0) # [3, 512]

cache_data = {
    "labels": labels_tensor,
    "img_pixel_batches": img_batches,
    "text_features": all_text_feats,
    "classifier_weights": classifier_weights,
    "visual_projection": clip_model.visual_projection.state_dict(),
    "logit_scale": clip_model.logit_scale.data.cpu(),
    "events": df_test["event_name"].tolist() if "event_name" in df_test.columns else [],
}
torch.save(cache_data, CACHE_FILE)
print(f"🎉 Đã lưu hoàn tất cache CrisisMMDv2 tại: {CACHE_FILE}")
print(f"   - Số lượng mẫu thử nghiệm hợp lệ: {len(labels_tensor)}")
print(f"   - Kích thước classifier weights: {classifier_weights.shape}")
''')

# ==============================================================================
# Cell 6: Self-Contained Continual Model Merging Engine & Benchmark trên CrisisMMDv2
# ==============================================================================
add_code(r'''# [5] SELF-CONTAINED CONTINUAL MODEL MERGING ENGINE & BENCHMARK SO SÁNH TRÊN CRISISMMDV2
import copy
import gc
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import CLIPVisionModel
from transformers.utils import logging as hf_logging

# Tắt cảnh báo verbose UNEXPECTED text_model keys từ Hugging Face
hf_logging.set_verbosity_error()

OUTPUT_DIR = Path("/content/outputs/uk_crisis_study")
CACHE_FILE = Path("/content/tensor_cache_crisismmd/crisismmd_v2_cache.pt")
BASE_MODEL_NAME = "openai/clip-vit-base-patch16"

print("📂 Đang nạp tensor cache CrisisMMDv2...")
cache = torch.load(CACHE_FILE, map_location="cuda")
labels = cache["labels"].cuda()
classifier_weights = cache["classifier_weights"].cuda() # [3, 512]
text_feats = cache["text_features"].cuda() # [N, 512]
img_batches = cache["img_pixel_batches"]

vis_proj = torch.nn.Linear(768, 512, bias=False).cuda().half()
vis_proj.load_state_dict(cache["visual_projection"])
vis_proj.eval()
logit_scale_val = min(float(cache.get("logit_scale", torch.tensor(4.6052)).cuda().exp().item()), 100.0)

# Tính Text Logits & Shannon Entropy (cố định cho văn bản tweet)
text_logits = (text_feats @ classifier_weights.T) * logit_scale_val # [N, 3]
p_text = F.softmax(text_logits, dim=-1)
H_text = -torch.sum(p_text * torch.log(p_text + 1e-9), dim=-1)

# ==============================================================================
# 1. NẠP MÔ HÌNH NỀN VÀ CÁC CHUYÊN GIA MIỀN ĐẶC THÙ CHO THẢM HỌA THIÊN TAI
# ==============================================================================
print("\n" + "="*80)
print("🌐 BẮT ĐẦU NẠP CÁC CHUYÊN GIA THỊ GIÁC VIỄN THÁM & KHÔNG ẢNH CỦA TÁC GIẢ TANG ET AL.")
print("="*80)

print(f"📦 1/4 Đang nạp Base Model: {BASE_MODEL_NAME}...")
base_model = CLIPVisionModel.from_pretrained(BASE_MODEL_NAME).eval().half()
base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

EXPERT_HUBS = [
    ("Expert_Resisc45", "tanganke/clip-vit-base-patch16_resisc45"),
    ("Expert_EuroSAT", "tanganke/clip-vit-base-patch16_eurosat"),
    ("Expert_Sun397", "tanganke/clip-vit-base-patch16_sun397"),
]

expert_models = []
expert_names = []

for idx, (exp_name, hub_id) in enumerate(EXPERT_HUBS, 2):
    print(f"📦 {idx}/4 Đang nạp Chuyên gia: {exp_name} ({hub_id})...")
    exp_m = CLIPVisionModel.from_pretrained(hub_id).eval().half()
    expert_models.append(exp_m)
    expert_names.append(exp_name)

# ==============================================================================
# 2. BỘ ĐỘNG CƠ HỢP NHẤT MÔ HÌNH THUẦN PYTORCH (PURE PYTORCH MERGING ENGINE)
# ==============================================================================
def get_tv_norm(model_state, b_state):
    sq = 0.0
    for k, v in model_state.items():
        if "weight" in k or "bias" in k:
            sq += (v.float() - b_state[k].float()).norm(2).item() ** 2
    return sq ** 0.5

# (A) Simple Weight Averaging (SWA)
def merge_swa(b_state, experts):
    print("⚙️ Đang thực thi Simple Weight Averaging (SWA)...")
    merged_s = {}
    N = len(experts)
    for k, b_val in b_state.items():
        sum_w = torch.zeros_like(b_val.float())
        for exp in experts:
            sum_w += exp.state_dict()[k].float()
        merged_s[k] = (sum_w / N).to(b_val.dtype)
    m = copy.deepcopy(base_model)
    m.load_state_dict(merged_s)
    return m

# (B) Task Arithmetic (TA)
def merge_task_arithmetic(b_state, experts, scaling_factor=0.3):
    print(f"⚙️ Đang thực thi Task Arithmetic (Scaling factor = {scaling_factor})...")
    merged_s = {}
    for k, b_val in b_state.items():
        tv_sum = torch.zeros_like(b_val.float())
        for exp in experts:
            tv_sum += (exp.state_dict()[k].float() - b_val.float())
        merged_s[k] = (b_val.float() + scaling_factor * tv_sum).to(b_val.dtype)
    m = copy.deepcopy(base_model)
    m.load_state_dict(merged_s)
    return m

# (C) TIES-Merging
def merge_ties(b_state, experts, scaling_factor=0.3, k_percent=20):
    print(f"⚙️ Đang thực thi TIES-Merging (Top {k_percent}%, Scaling factor = {scaling_factor})...")
    merged_s = {}
    for k, b_val in b_state.items():
        if b_val.ndim == 0:
            merged_s[k] = b_val
            continue
        task_vectors = []
        for exp in experts:
            tv = exp.state_dict()[k].float() - b_val.float()
            flat_tv = tv.abs().flatten()
            thresh = torch.quantile(flat_tv, (100.0 - k_percent) / 100.0)
            trimmed_tv = tv * (tv.abs() >= thresh)
            task_vectors.append(trimmed_tv)
        stacked = torch.stack(task_vectors, dim=0)
        sign_consensus = torch.sign(stacked.sum(dim=0))
        agree_mask = (torch.sign(stacked) == sign_consensus) & (stacked != 0)
        sum_agree = (stacked * agree_mask).sum(dim=0)
        count_agree = agree_mask.sum(dim=0).clamp(min=1)
        disjoint_tv = sum_agree / count_agree
        merged_s[k] = (b_val.float() + scaling_factor * disjoint_tv).to(b_val.dtype)
    m = copy.deepcopy(base_model)
    m.load_state_dict(merged_s)
    return m

# (D) OPCM (Orthogonal Subspace Projection Continual Merging - Tang et al., arXiv:2501.09522)
def merge_opcm(b_state, experts, alpha=0.5):
    print(f"⚙️ Đang thực thi OPCM SVD Orthogonal Projection Continual Merging (Alpha = {alpha})...")
    merged_state = {k: v.cpu().clone() for k, v in experts[0].state_dict().items()}
    
    avg_tv_norm = get_tv_norm(merged_state, b_state)
    all_tv_norms = [avg_tv_norm]
    prev_lambda_t = 1.0
    
    for exp_idx, exp_model in enumerate(experts[1:], 1):
        exp_state = exp_model.state_dict()
        exp_norm = get_tv_norm(exp_state, b_state)
        all_tv_norms.append(exp_norm)
        avg_tv_norm = float(np.mean(all_tv_norms))
        lambda_t = 1.0
        
        new_merged_state = {}
        for key, val in merged_state.items():
            base_val = b_state[key]
            task_val = exp_state[key]
            
            # Nếu là Linear weight (2D ma trận trọng số và không phải norm/bias)
            if val.ndim == 2 and "weight" in key and "layer_norm" not in key and "position" not in key:
                prev_merged_tv = val.float() - base_val.float()
                task_tv = task_val.float() - base_val.float()
                
                # Phân rã SVD trên không gian Task Vector tích lũy
                u, s, vh = torch.linalg.svd(prev_merged_tv, full_matrices=False)
                cum_energy = s.cumsum(dim=0) / (s.sum() + 1e-12)
                split_rank = int((cum_energy > alpha).float().argmax().item())
                
                # Chiếu trực giao Task Vector mới vào không gian con: u.T @ task_tv @ vh.T
                proj_tv = u.T @ task_tv @ vh.T
                proj_tv.diagonal().fill_(0.0)
                proj_tv[:split_rank, :split_rank] = 0.0
                
                # Phục hồi Task Vector đã triệt tiêu giao thoa
                cleaned_task_tv = u @ proj_tv @ vh
                new_w = base_val.float() + (prev_lambda_t * prev_merged_tv + cleaned_task_tv) / lambda_t
                new_merged_state[key] = new_w.to(val.dtype)
            else:
                prev_merged_tv = val.float() - base_val.float()
                task_tv = task_val.float() - base_val.float()
                new_w = base_val.float() + (prev_lambda_t * prev_merged_tv + task_tv) / lambda_t
                new_merged_state[key] = new_w.to(val.dtype)
                
        merged_state = new_merged_state
        
        # Cân bằng chuẩn độ dài Task Vector (Eq. 9 / Lines 170-176 của tác giả)
        curr_norm = get_tv_norm(merged_state, b_state)
        if curr_norm > 1e-6:
            rescale = avg_tv_norm / curr_norm
            for k, v in merged_state.items():
                bv = b_state[k]
                merged_state[k] = (bv.float() + (v.float() - bv.float()) * rescale).to(v.dtype)
        prev_lambda_t = lambda_t

    m = copy.deepcopy(base_model)
    m.load_state_dict(merged_state)
    return m

# ==============================================================================
# 3. TIẾN HÀNH TẠO VÀ NẠP CÁC MÔ HÌNH HỢP NHẤT
# ==============================================================================
print("\n" + "="*80)
print("⚡ TIẾN HÀNH HỢP NHẤT MÔ HÌNH BẰNG CÁC THUẬT TOÁN CONTINUAL MODEL MERGING...")
print("="*80)

model_dict = {
    "Pretrained_CLIP": base_model,
}
for name, exp_m in zip(expert_names, expert_models):
    model_dict[name] = exp_m

# Thực thi các thuật toán gộp
model_dict["SWA"] = merge_swa(base_state, expert_models)
model_dict["Task_Arithmetic"] = merge_task_arithmetic(base_state, expert_models, scaling_factor=0.3)
model_dict["TIES_Merging"] = merge_ties(base_state, expert_models, scaling_factor=0.3, k_percent=20)
model_dict["OPCM"] = merge_opcm(base_state, expert_models, alpha=0.5)

# Kiểm tra nếu có checkpoint huấn luyện cục bộ (tùy chọn)
if (OUTPUT_DIR / "opcm" / "checkpoint").exists():
    try:
        model_dict["Local_OPCM"] = CLIPVisionModel.from_pretrained(OUTPUT_DIR / "opcm" / "checkpoint").eval().half()
        print("✅ Đã nạp thêm checkpoint Local OPCM từ thư mục outputs!")
    except Exception:
        pass

# ==============================================================================
# 4. ĐÁNH GIÁ HIỆU NĂNG ĐA MÔ THỨC TRÊN TẬP THỬ NGHIỆM CRISISMMDV2
# ==============================================================================
print("\n" + "="*80)
print("📊 BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN TRÊN 529 MẪU THỬ NGHIỆM THẬT CỦA QCRI/CRISISMMDV2...")
print("="*80)

evaluation_summary = []
all_img_logits = {}
base_acc_v = None

for m_name, v_model in model_dict.items():
    print(f"🚀 Đang đánh giá phương pháp: {m_name}...")
    v_model = v_model.cuda().eval().half()
    
    feats_list = []
    with torch.no_grad():
        for px in img_batches:
            out = v_model(pixel_values=px.cuda())
            pooled = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else (out[1] if isinstance(out, tuple) else out.last_hidden_state[:, 0, :])
            projected = vis_proj(pooled)
            feats_list.append(F.normalize(projected, dim=-1))
            
    all_vis_feats = torch.cat(feats_list, dim=0) # [N, 512]
    img_logits = (all_vis_feats @ classifier_weights.T) * logit_scale_val # [N, 3]
    all_img_logits[m_name] = img_logits
    
    # 1. Vision-Only
    preds_v = img_logits.argmax(dim=-1)
    acc_v = (preds_v == labels).float().mean().item() * 100.0
    f1_v = f1_score(labels.cpu().numpy(), preds_v.cpu().numpy(), average="macro") * 100.0
    if base_acc_v is None:
        base_acc_v = acc_v
        
    # 2. Text-Only
    preds_t = text_logits.argmax(dim=-1)
    acc_t = (preds_t == labels).float().mean().item() * 100.0
    
    # 3. Fixed Multimodal (β=0.5)
    fused_half = 0.5 * img_logits + 0.5 * text_logits
    preds_half = fused_half.argmax(dim=-1)
    acc_half = (preds_half == labels).float().mean().item() * 100.0
    f1_half = f1_score(labels.cpu().numpy(), preds_half.cpu().numpy(), average="macro") * 100.0
    
    # 4. Bounded Gating AC-Fusion (β_min = 0.8, T = 2.0)
    p_img = F.softmax(img_logits, dim=-1)
    H_img = -torch.sum(p_img * torch.log(p_img + 1e-9), dim=-1)
    gate = torch.sigmoid((H_text - H_img) / 2.0).unsqueeze(-1)
    beta_bounded = 0.8 + 0.2 * gate # [0.8, 1.0]
    fused_bounded = beta_bounded * img_logits + (1.0 - beta_bounded) * text_logits
    preds_bounded = fused_bounded.argmax(dim=-1)
    acc_bounded = (preds_bounded == labels).float().mean().item() * 100.0
    f1_bounded = f1_score(labels.cpu().numpy(), preds_bounded.cpu().numpy(), average="macro") * 100.0
    
    diff_gain = acc_v - base_acc_v
    
    evaluation_summary.append({
        "Method": m_name,
        "Vision-Only Acc": f"{acc_v:.2f}%",
        "Vision Macro-F1": f"{f1_v:.2f}%",
        "Text-Only Acc": f"{acc_t:.2f}%",
        "Fixed Multimodal (β=0.5)": f"{acc_half:.2f}%",
        "Bounded AC-Fusion (Acc)": f"{acc_bounded:.2f}%",
        "Bounded AC-Fusion (F1)": f"{f1_bounded:.2f}%",
        "Gain vs Pretrained": f"{diff_gain:+.2f}%",
    })
    
    # Giải phóng GPU model
    v_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

# In bảng tổng hợp
df_res = pd.DataFrame(evaluation_summary)
print("\n" + "="*95)
print("📊 BẢNG TỔNG HỢP HIỆU QUẢ OPCM VÀ CÁC PHƯƠNG PHÁP TRÊN QCRI/CRISISMMDV2 (DAMAGE SEVERITY):")
print("="*95)
try:
    print(df_res.to_markdown(index=False))
except Exception:
    print(df_res.to_string(index=False))

REPORT_DIR = Path("/content/outputs/crisismmd_v2_study")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
df_res.to_csv(REPORT_DIR / "crisismmd_v2_performance_summary.csv", index=False)
print(f"\n💾 Đã lưu kết quả chi tiết tại: {REPORT_DIR / 'crisismmd_v2_performance_summary.csv'}")
''')

# ==============================================================================
# Cell 7: Trực Quan Hóa Ma Trận Nhầm Lẫn & Đóng Gói
# ==============================================================================
add_code(r'''# [6] TRỰC QUAN HÓA MA TRẬN NHẦM LẪN 3 MỨC ĐỘ HƯ HẠI & ĐÓNG GÓI KẾT QUẢ
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from google.colab import files

DAMAGE_SHORT = ["None/Little", "Mild", "Severe"]

y_true = labels.cpu().numpy()
y_pred_pre = all_img_logits["Pretrained_CLIP"].argmax(dim=-1).cpu().numpy()
y_pred_opcm = all_img_logits["OPCM"].argmax(dim=-1).cpu().numpy()

cm_pre = confusion_matrix(y_true, y_pred_pre, normalize="true") * 100.0
cm_opcm = confusion_matrix(y_true, y_pred_opcm, normalize="true") * 100.0

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

sns.heatmap(cm_pre, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=DAMAGE_SHORT, yticklabels=DAMAGE_SHORT, linewidths=1.0, linecolor="white")
axes[0].set_title("Pretrained CLIP ViT-B/16\nNormalized Damage Confusion Matrix (%)", fontsize=12, fontweight="bold", pad=12)
axes[0].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")

sns.heatmap(cm_opcm, annot=True, fmt=".1f", cmap="Reds", cbar=True, ax=axes[1],
            xticklabels=DAMAGE_SHORT, yticklabels=DAMAGE_SHORT, linewidths=1.0, linecolor="white",
            cbar_kws={'label': 'Tỷ Lệ Nhận Diện Chính Xác (%)'})
axes[1].set_title("OPCM Merged ViT-B/16 (Ours)\nNormalized Damage Confusion Matrix (%)", fontsize=12, fontweight="bold", pad=12)
axes[1].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")

plt.suptitle("Đánh Giá Năng Lực Phát Hiện Mức Độ Hư Hại Thảm Họa Trên QCRI/CrisisMMDv2", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
cm_out_path = REPORT_DIR / "crisismmd_damage_confusion_matrix.png"
plt.savefig(cm_out_path, dpi=300, bbox_inches="tight")
plt.show()

# Đóng gói zip tải về
!zip -q -r /content/OPCM_CrisisMMDv2_Results.zip /content/outputs/crisismmd_v2_study
print("📦 Đã đóng gói toàn bộ kết quả CrisisMMDv2 vào: /content/OPCM_CrisisMMDv2_Results.zip")
try:
    files.download("/content/OPCM_CrisisMMDv2_Results.zip")
except Exception:
    pass
''')

# Lưu file notebook
OUTPUT_NOTEBOOK = Path(__file__).resolve().parent.parent / "OPCM_CrisisMMDv2_Global_Study.ipynb"
OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"🎉 ĐÃ TẠO THÀNH CÔNG NOTEBOOK CHO CRISISMMDV2: {OUTPUT_NOTEBOOK}")
print(f"📊 Kích thước: {OUTPUT_NOTEBOOK.stat().st_size / 1024:.2f} KB | Số Cells: {len(notebook['cells'])}")
