"""Regenerate the checksum-verified continual-baseline patch inside the Colab notebook."""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MM_HOPE_Colab_Pro_GPU_Study.ipynb"
PATCH_FILES = (
    "experiments/mm_hope/requirements-colab-gpu.txt",
    "experiments/mm_hope/baseline_preflight.py",
    "experiments/mm_hope/continual_baselines.py",
    "experiments/mm_hope/continual_streams.py",
    "experiments/mm_hope/frozen_baseline_runner.py",
    "experiments/mm_hope/fusion.py",
    "experiments/mm_hope/gpu_study_cli.py",
    "experiments/mm_hope/gpu_study_models.py",
    "experiments/mm_hope/gpu_study_runner.py",
    "experiments/mm_hope/gpu_study_trainer.py",
)
BEGIN = "# BEGIN CONTINUAL BASELINE PATCH BUNDLE\n"
END = "# END CONTINUAL BASELINE PATCH BUNDLE\n"
RUN_BEGIN = "# BEGIN CONTINUAL BASELINE DEVELOPMENT\n"
RUN_END = "# END CONTINUAL BASELINE DEVELOPMENT\n"
EXPORT_BEGIN = "# BEGIN MULTI-ROOT RESULT EXPORT\n"
EXPORT_END = "# END MULTI-ROOT RESULT EXPORT\n"


def _archive_payload() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in PATCH_FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _wrapped_base64(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = [encoded[index : index + 100] for index in range(0, len(encoded), 100)]
    return "\n".join(f"    '{chunk}'" for chunk in chunks)


def _replace_marked_block(source: str, begin: str, end: str, replacement: str) -> str:
    start = source.find(begin)
    if start >= 0:
        finish = source.find(end, start)
        if finish < 0:
            raise RuntimeError(f"missing closing marker: {end.strip()}")
        source = source[:start] + source[finish + len(end) :]
    return source.rstrip("\n") + "\n" + replacement.lstrip("\n")


def _patch_bundle_block(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return (
        "\n"
        + BEGIN
        + "CONTINUAL_BUNDLE_SHA256 = '"
        + digest
        + "'\n"
        + "CONTINUAL_BUNDLE_B64 = (\n"
        + _wrapped_base64(payload)
        + "\n)\n"
        + "continual_payload = base64.b64decode(CONTINUAL_BUNDLE_B64)\n"
        + "assert hashlib.sha256(continual_payload).hexdigest() == CONTINUAL_BUNDLE_SHA256\n"
        + "with zipfile.ZipFile(io.BytesIO(continual_payload)) as continual_zip:\n"
        + "    continual_zip.extractall(PROJECT_ROOT)\n"
        + "del CONTINUAL_BUNDLE_B64, continual_payload\n"
        + END
    )


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", [])).replace("\r\n", "\n")


def _set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _replace_assignment(source: str, name: str, value: str) -> str:
    prefix = f"{name} = "
    start = source.find(prefix)
    if start < 0:
        raise RuntimeError(f"notebook assignment is missing: {name}")
    finish = source.find("\n", start)
    if finish < 0:
        raise RuntimeError(f"notebook assignment is incomplete: {name}")
    return source[:start] + f"{name} = {value}" + source[finish:]


def _update_configuration(source: str) -> str:
    legacy_dataset_url = (
        "# Public UK-CrisisMMD archive supplied by the dataset owner.\n"
        "# The stable file-ID URL avoids expiring uuid/at query parameters.\n"
        "UK_DATASET_URL = ('https://drive.usercontent.google.com/download?'\n"
        "                  'id=1SIbVIKp_mCsageSqqOu-D-jlCnlleOJj&export=download&confirm=t')\n\n"
    )
    source = source.replace(legacy_dataset_url, "")
    source = source.replace(
        "# Test-only confirmation: lock beta=0 and beta=0.50 development artifacts, then run five projection seeds.\n",
        "# Test-only confirmation: lock beta=0 and beta=0.50 development artifacts, then run ten projection seeds.\n",
    )
    if "UK_DATASET_FILE_ID" not in source:
        needle = "\n# quick: 1-epoch subset with G01 on both datasets (pipeline check)\n"
        if needle not in source:
            raise RuntimeError("notebook dataset configuration changed unexpectedly")
        source = source.replace(
            needle,
            "\n# Owner-provided file; Google Drive confirmation tokens are resolved at run time.\n"
            + "UK_DATASET_FILE_ID = '1SIbVIKp_mCsageSqqOu-D-jlCnlleOJj'\n"
            + "# Optional fallbacks: use a direct public mirror URL or an archive already uploaded to /content.\n"
            + "UK_DATASET_MIRROR_URL = ''\n"
            + "UK_DATASET_LOCAL_PATH = ''\n"
            + needle,
            1,
        )
    if "# baseline_only:" not in source:
        needle = "# full: development -> immutable lock -> 5-seed test confirmation -> inference\n"
        if needle not in source:
            raise RuntimeError("notebook preset documentation changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# baseline_only: skip G01--G06 static training and run only the enabled continual-baseline stages\n",
        )
    source = source.replace(
        "assert RUN_PRESET in {'quick', 'development', 'full'}",
        "assert RUN_PRESET in {'quick', 'development', 'full', 'baseline_only'}",
    )
    if "RUN_CONTINUAL_BASELINES" not in source:
        needle = "DOWNLOAD_RESULTS_AT_END = True\n"
        if needle not in source:
            raise RuntimeError("notebook configuration cell changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Set true to run the validation-only R0--R3 S-Frozen suite after the selected study stage.\n"
            + "RUN_CONTINUAL_BASELINES = True\n"
            + "CONTINUAL_METHODS = ['B_ER', 'R0_F', 'R0_NCM', 'R1_CA', 'R2_H', 'R3_U', 'R3_W']\n",
        )
    if "RUN_CONTINUAL_CONFIRMATION" not in source:
        needle = "CONTINUAL_METHODS = ['B_ER', 'R0_F', 'R0_NCM', 'R1_CA', 'R2_H', 'R3_U', 'R3_W']\n"
        if needle not in source:
            raise RuntimeError("notebook continual configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Confirmation accesses the test split; leave disabled until development results are reviewed and locked.\n"
            + "RUN_CONTINUAL_CONFIRMATION = False\n"
            + "CONTINUAL_CONFIRMATION_SEEDS = [2027, 2028, 2029, 2030, 2031]\n",
        )
    if "RUN_RANPAC_STABILITY" not in source:
        needle = "CONTINUAL_CONFIRMATION_SEEDS = [2027, 2028, 2029, 2030, 2031]\n"
        if needle not in source:
            raise RuntimeError("notebook confirmation configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Runs four validation-only RanPAC variants; it never reads the test split.\n"
            + "RUN_RANPAC_STABILITY = False\n",
        )
    if "RUN_RANPAC_MECHANISM_ABLATION" not in source:
        needle = "RUN_RANPAC_STABILITY = False\n"
        if needle not in source:
            raise RuntimeError("notebook RanPAC stability configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Runs six validation-only controls to isolate projection dimension, normalization, ridge, and solver precision.\n"
            + "RUN_RANPAC_MECHANISM_ABLATION = False\n",
        )
    if "RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT" not in source:
        needle = next(
            (
                value
                for value in (
                    "RUN_RANPAC_MECHANISM_ABLATION = False\n",
                    "RUN_RANPAC_MECHANISM_ABLATION = True\n",
                )
                if value in source
            ),
            None,
        )
        if needle is None:
            raise RuntimeError("notebook mechanism-ablation configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Runs four validation-only train-prior score adjustments; it never reads the test split.\n"
            + "RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT = False\n",
        )
    if "RUN_RANPAC_CLASS_BALANCED_RIDGE" not in source:
        source = source.replace(
            "RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT = True\n",
            "RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT = False\n",
        )
        needle = "RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT = False\n"
        if needle not in source:
            raise RuntimeError("notebook class-prior configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Runs four validation-only class-balanced RanPAC ridge fits; it never reads the test split.\n"
            + "RUN_RANPAC_CLASS_BALANCED_RIDGE = True\n",
        )
    if "RUN_RANPAC_CLASS_BALANCED_CONFIRMATION" not in source:
        needle = "RUN_RANPAC_CLASS_BALANCED_RIDGE = True\n"
        if needle not in source:
            needle = "RUN_RANPAC_CLASS_BALANCED_RIDGE = False\n"
        if needle not in source:
            raise RuntimeError("notebook class-balanced confirmation configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Test-only confirmation: lock beta=0 and beta=0.50 development artifacts, then run ten projection seeds.\n"
            + "RUN_RANPAC_CLASS_BALANCED_CONFIRMATION = False\n",
        )
    if "RANPAC_CLASS_BALANCED_CONFIRMATION_SEEDS" not in source:
        needle = next(
            (
                value
                for value in (
                    "RUN_RANPAC_CLASS_BALANCED_CONFIRMATION = False\n",
                    "RUN_RANPAC_CLASS_BALANCED_CONFIRMATION = True\n",
                )
                if value in source
            ),
            None,
        )
        if needle is None:
            raise RuntimeError("notebook class-balanced confirmation flag changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "RANPAC_CLASS_BALANCED_CONFIRMATION_SEEDS = "
            + "[2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036]\n",
            1,
        )
    if "RANPAC_CLASS_BALANCED_CANDIDATES" not in source:
        needle = next(
            (
                value
                for value in (
                    "RUN_RANPAC_CLASS_BALANCED_RIDGE = True\n",
                    "RUN_RANPAC_CLASS_BALANCED_RIDGE = False\n",
                )
                if value in source
            ),
            None,
        )
        if needle is None:
            raise RuntimeError("notebook class-balanced ridge configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Run the complete class-balanced ridge grid; completed result.json files resume automatically.\n"
            + "RANPAC_CLASS_BALANCED_CANDIDATES = ['balanced_p1024_rel1e-1_beta0p25_cpu_fp64', 'balanced_p1024_rel1e-1_beta0p50_cpu_fp64', 'balanced_p1024_rel1e-1_beta0p75_cpu_fp64', 'balanced_p1024_rel1e-1_beta1p00_cpu_fp64']\n"
            + "RANPAC_CLASS_BALANCED_DATASETS = ['uk_crisis_type', 'crisismmd_humanitarian']\n",
        )
    if "RANPAC_CLASS_BALANCED_DATASETS" not in source:
        start = source.find("RANPAC_CLASS_BALANCED_CANDIDATES = ")
        if start < 0:
            raise RuntimeError("notebook class-balanced candidate configuration changed unexpectedly")
        finish = source.find("\n", start)
        if finish < 0:
            raise RuntimeError("notebook class-balanced candidate line is incomplete")
        source = (
            source[: finish + 1]
            + "RANPAC_CLASS_BALANCED_DATASETS = ['uk_crisis_type', 'crisismmd_humanitarian']\n"
            + source[finish + 1 :]
        )
    source = _replace_assignment(
        source,
        "RANPAC_CLASS_BALANCED_CANDIDATES",
        "['balanced_p1024_rel1e-1_beta0p00_cpu_fp64', 'balanced_p1024_rel1e-1_beta0p25_cpu_fp64', 'balanced_p1024_rel1e-1_beta0p50_cpu_fp64', 'balanced_p1024_rel1e-1_beta0p75_cpu_fp64', 'balanced_p1024_rel1e-1_beta1p00_cpu_fp64']",
    )
    source = source.replace(
        "# Runs four validation-only class-balanced RanPAC ridge fits; it never reads the test split.\n",
        "# Runs five validation-only class-balanced RanPAC ridge fits; it never reads the test split.\n",
    )
    if "COLAB_SAFE_MODE" not in source:
        needle = "DELETE_ARCHIVES_AFTER_EXTRACTION = True\n"
        if needle not in source:
            raise RuntimeError("notebook archive configuration changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "# Resource-only controls: they do not alter labels, splits, or estimator equations.\n"
            + "COLAB_SAFE_MODE = True\n"
            + "FROZEN_FEATURE_BATCH_SIZE = 8 if COLAB_SAFE_MODE else 32\n"
            + "FEATURE_NUM_WORKERS = 0 if COLAB_SAFE_MODE else 2\n"
            + "STATIC_NUM_WORKERS = 0 if COLAB_SAFE_MODE else 2\n",
        )
    if "assert FROZEN_FEATURE_BATCH_SIZE > 0" not in source:
        needle = "assert RUN_PRESET in {'quick', 'development', 'full', 'baseline_only'}\n"
        if needle not in source:
            raise RuntimeError("notebook preset assertion changed unexpectedly")
        source = source.replace(
            needle,
            needle
            + "assert FROZEN_FEATURE_BATCH_SIZE > 0\n"
            + "assert FEATURE_NUM_WORKERS >= 0 and STATIC_NUM_WORKERS >= 0\n",
        )
    return source


def _update_system_setup(source: str) -> str:
    if "PYTORCH_CUDA_ALLOC_CONF" not in source:
        needle = "\nimport torch\n"
        if needle not in source:
            raise RuntimeError("notebook torch setup changed unexpectedly")
        source = source.replace(
            needle,
            "\n# Configure the allocator before torch initializes CUDA.\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
            "os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')\n"
            "os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')\n"
            + needle,
            1,
        )
    old = (
        "subprocess.run(['apt-get', 'update', '-qq'], check=True)\n"
        "subprocess.run(['apt-get', 'install', '-y', '-qq', 'aria2', 'unzip'], check=True)\n"
    )
    if old in source:
        source = source.replace(
            old,
            "if shutil.which('aria2c') is None or shutil.which('unzip') is None:\n"
            "    subprocess.run(['apt-get', 'update', '-qq'], check=True)\n"
            "    subprocess.run(['apt-get', 'install', '-y', '-qq', 'aria2', 'unzip'], check=True)\n"
            "else:\n"
            "    print('aria2 and unzip are already installed; skipping apt.')\n",
            1,
        )
    return source


def _safe_download_cell() -> str:
    return """# 3. Fast verified downloads and extraction
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UK_DATASET_NAME = 'UK_CrisisMMD_Public.zip'
UK_DATASET_SIZE = 1_140_635_966
UK_DATASET_SHA256 = '77f080a37a21eed5c0b044bb3c729bdb80818ac94b8d40e64c412b5664b90b53'
CRISIS_MEDIA_URL = 'https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz'
CRISIS_MEDIA_SHA256 = '351b9be6123a5d5074407d792bfdddc1f470d4909f65ddd429b584d24607d43f'
CRISIS_LABEL_URL = 'https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip'
CRISIS_LABEL_SHA256 = '32f2203bbf0545057be2c6d5c95636cfb5032aef58d0f57e04c5dfac246843ff'

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

class GoogleDriveDownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self._inside_download_form = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'form' and attributes.get('id') == 'download-form':
            self.action = attributes.get('action')
            self._inside_download_form = True
        elif tag == 'input' and self._inside_download_form:
            name = attributes.get('name')
            if name:
                self.fields[name] = attributes.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form' and self._inside_download_form:
            self._inside_download_form = False

def resolve_google_drive_download(file_id):
    landing_url = 'https://drive.google.com/uc?' + urlencode(
        {'export': 'download', 'id': file_id}
    )
    request = Request(landing_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return response.geturl()
        page = response.read(1024 * 1024).decode('utf-8', errors='replace')

    parser = GoogleDriveDownloadForm()
    parser.feed(page)
    if not parser.action or not parser.fields.get('uuid'):
        title = 'Google Drive response'
        lowered = page.lower()
        if 'quota exceeded' in lowered or 'download quota' in lowered:
            title = 'Google Drive download quota is exceeded'
        raise RuntimeError(
            title + '. Set UK_DATASET_MIRROR_URL to a fresh direct mirror, or upload the '
            'verified ZIP to /content and set UK_DATASET_LOCAL_PATH.'
        )
    direct_url = parser.action + '?' + urlencode(parser.fields)
    print('Resolved a fresh Google Drive confirmation token.')
    return direct_url

def _html_download_error(path):
    with Path(path).open('rb') as handle:
        prefix = handle.read(8192).decode('utf-8', errors='ignore').lower()
    if '<html' not in prefix and '<!doctype html' not in prefix:
        return None
    if 'quota exceeded' in prefix or 'download quota' in prefix:
        return 'Google Drive download quota is exceeded'
    if 'virus scan warning' in prefix:
        return 'Google Drive returned an unresolved confirmation page'
    return 'the server returned HTML instead of an archive'

def _discard_download(path):
    path = Path(path)
    path.unlink(missing_ok=True)
    Path(str(path) + '.aria2').unlink(missing_ok=True)

def aria2_download(url, destination, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    control_file = Path(str(destination) + '.aria2')
    if destination.is_file():
        if expected_size is None or destination.stat().st_size == expected_size:
            print('Reusing downloaded archive:', destination.name)
            return destination
        problem = _html_download_error(destination)
        if control_file.is_file() and problem is None:
            print(
                'Resuming partial download:', destination.name,
                f'({destination.stat().st_size}/{expected_size} bytes)',
            )
        else:
            print(
                'Discarding invalid cached download:', destination.name,
                f'({destination.stat().st_size} bytes' + (f'; {problem}' if problem else '') + ')',
            )
            _discard_download(destination)
    elif control_file.is_file():
        control_file.unlink()
    command = [
        'aria2c', '--continue=true', '--max-connection-per-server=16',
        '--split=16', '--min-split-size=1M', '--file-allocation=none',
        '--max-tries=5', '--retry-wait=5', '--timeout=60',
        '--auto-file-renaming=false', '--allow-overwrite=true',
        '--header=User-Agent: Mozilla/5.0',
        '--dir', str(destination.parent), '--out', destination.name, url,
    ]
    print('Downloading:', destination.name)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        hint = (
            'Set UK_DATASET_MIRROR_URL to a fresh direct mirror, or upload the '
            'verified ZIP to /content and set UK_DATASET_LOCAL_PATH.'
            if destination.name == UK_DATASET_NAME
            else 'The source endpoint may be temporarily unavailable or throttled.'
        )
        raise RuntimeError(
            f'aria2 could not download {destination.name}. {hint}'
        ) from error
    if expected_size is not None and destination.stat().st_size != expected_size:
        observed_size = destination.stat().st_size
        problem = _html_download_error(destination)
        _discard_download(destination)
        detail = f' ({problem})' if problem else ''
        raise RuntimeError(
            f'{destination.name} download is {observed_size} bytes; expected '
            f'{expected_size}{detail}. Set UK_DATASET_MIRROR_URL to a fresh direct '
            'mirror, or upload the verified ZIP to /content and set '
            'UK_DATASET_LOCAL_PATH.'
        )
    return destination

def verify(path, expected):
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f'SHA-256 mismatch for {path.name}: {observed}')
    print('Verified:', path.name, observed)

crisis_root = PROJECT_ROOT / 'external_data' / 'crisismmd_v2'
media_root = crisis_root / 'media'
label_root = crisis_root / 'crisismmd_datasplit_agreed_label'
required = [
    PROJECT_ROOT / 'UK_CrisisMMD_Public' / 'public_dataset.jsonl',
    PROJECT_ROOT / 'analysis_outputs' / 'dataset_version_manifest.csv',
    PROJECT_ROOT / 'analysis_outputs' / 'joint_query_post_split_assignments.csv',
    PROJECT_ROOT / 'analysis_outputs' / 'image_content_quality_features.csv',
    media_root / 'CrisisMMD_v2.0' / 'data_image',
    label_root / 'crisismmd_datasplit_agreed_label' / 'task_humanitarian_text_img_agreed_lab_train.tsv',
]

if all(path.exists() for path in required):
    print('Extracted datasets already exist; skipping download and extraction.')
else:
    if UK_DATASET_LOCAL_PATH:
        uk_archive = Path(UK_DATASET_LOCAL_PATH).expanduser()
        if not uk_archive.is_file():
            raise FileNotFoundError(f'UK_DATASET_LOCAL_PATH does not exist: {uk_archive}')
        if uk_archive.stat().st_size != UK_DATASET_SIZE:
            raise RuntimeError(
                f'Local UK archive is {uk_archive.stat().st_size} bytes; expected '
                f'{UK_DATASET_SIZE}'
            )
        print('Using locally uploaded UK archive:', uk_archive)
    else:
        uk_url = UK_DATASET_MIRROR_URL.strip() or resolve_google_drive_download(
            UK_DATASET_FILE_ID
        )
        uk_archive = aria2_download(
            uk_url, DOWNLOAD_DIR / UK_DATASET_NAME, expected_size=UK_DATASET_SIZE
        )
    verify(uk_archive, UK_DATASET_SHA256)
    crisis_archive = aria2_download(CRISIS_MEDIA_URL, DOWNLOAD_DIR / 'CrisisMMD_v2.0.tar.gz')
    crisis_labels = aria2_download(
        CRISIS_LABEL_URL, DOWNLOAD_DIR / 'crisismmd_datasplit_agreed_label.zip'
    )
    verify(crisis_archive, CRISIS_MEDIA_SHA256)
    verify(crisis_labels, CRISIS_LABEL_SHA256)

    subprocess.run(['unzip', '-q', '-o', str(uk_archive), '-d', str(PROJECT_ROOT)], check=True)
    media_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xzf', str(crisis_archive), '-C', str(media_root)], check=True)
    label_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['unzip', '-q', '-o', str(crisis_labels), '-d', str(label_root)], check=True)
    if DELETE_ARCHIVES_AFTER_EXTRACTION:
        for archive in (uk_archive, crisis_archive, crisis_labels):
            archive.unlink(missing_ok=True)

missing = [str(path) for path in required if not path.exists()]
assert not missing, 'Extraction is incomplete: ' + repr(missing)

patch_path = PROJECT_ROOT / 'colab_bundle' / 'nested_learning_mm_hope.patch'
apply_check = subprocess.run(
    ['git', '-C', str(NESTED_REPO), 'apply', '--check', str(patch_path)],
    capture_output=True, text=True,
)
if apply_check.returncode == 0:
    subprocess.run(['git', '-C', str(NESTED_REPO), 'apply', str(patch_path)], check=True)
    print('Applied the pinned MM-HOPE correctness patch.')
else:
    reverse_check = subprocess.run(
        ['git', '-C', str(NESTED_REPO), 'apply', '--reverse', '--check', str(patch_path)],
        capture_output=True, text=True,
    )
    if reverse_check.returncode != 0:
        raise RuntimeError('Nested Learning patch cannot be applied cleanly: ' + apply_check.stderr)
    print('MM-HOPE correctness patch is already applied; continuing safely.')
print('Datasets and patched source are ready.')
"""


def _safe_install_cell() -> str:
    return """# 4. Install the pinned Python environment
requirements = PROJECT_ROOT / 'experiments' / 'mm_hope' / 'requirements-colab-gpu.txt'
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '-r', str(requirements)],
    check=True,
)

# Use an isolated Pillow wheel. Replacing Colab's active system package in-place
# can pair Python files from one release with an already-loaded _imaging binary.
PILLOW_VERSION = '11.3.0'
runtime_package_root = PROJECT_ROOT / '.colab_runtime_packages'
pillow_target = runtime_package_root / f'pillow_{PILLOW_VERSION.replace(".", "_")}'
pillow_marker = pillow_target / '.complete'
if not pillow_marker.is_file():
    if pillow_target.exists():
        assert pillow_target.resolve().parent == runtime_package_root.resolve()
        shutil.rmtree(pillow_target)
    pillow_target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--no-deps',
            '--target', str(pillow_target), f'Pillow=={PILLOW_VERSION}',
        ],
        check=True,
    )
    pillow_marker.write_text(PILLOW_VERSION, encoding='utf-8')
for module_name in list(sys.modules):
    if module_name == 'PIL' or module_name.startswith('PIL.'):
        sys.modules.pop(module_name, None)
if str(pillow_target) in sys.path:
    sys.path.remove(str(pillow_target))
sys.path.insert(0, str(pillow_target))

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-e', str(NESTED_REPO), '--no-deps'],
    check=True,
)
nested_source = NESTED_REPO / 'src'
if str(nested_source) in sys.path:
    sys.path.remove(str(nested_source))
sys.path.insert(0, str(nested_source))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import nested_learning
import nested_learning.levels
resolved_nested_source = Path(nested_learning.__file__).resolve()
assert nested_source.resolve() in resolved_nested_source.parents, (
    f'Wrong nested_learning package imported: {resolved_nested_source}'
)
print('Nested Learning import:', resolved_nested_source)

import PIL
from PIL import Image
assert PIL.__version__ == PILLOW_VERSION, f'Unexpected Pillow version: {PIL.__version__}'
probe_image = Image.new('RGB', (2, 2), color='white')
probe_image.close()
print('Pillow import:', PIL.__version__, PIL.__file__)

from experiments.mm_hope.gpu_study_provenance import implementation_fingerprint
from experiments.mm_hope.gpu_study_trainer import hardware_inventory

print(json.dumps(hardware_inventory(torch.device('cuda')), indent=2))
print('Implementation:', implementation_fingerprint(PROJECT_ROOT))
"""


def _safe_validation_cell() -> str:
    return """# 5. Validate and print the dataset analysis used by training
import gc
import pandas as pd
from IPython.display import display
from experiments.mm_hope.gpu_study_data import dataset_inventory, load_gpu_study_dataset

inventories = {}
for dataset_key in SELECTED_DATASETS:
    dataset = load_gpu_study_dataset(PROJECT_ROOT, dataset_key)
    distribution = train_counts = None
    try:
        inventory = dataset_inventory(dataset)
        inventories[dataset_key] = inventory
        print('\\nDATASET:', dataset_key)
        print('Rows:', inventory['rows'], '| Groups:', inventory['independent_groups'])
        print('Fingerprint:', inventory['fingerprint'])
        distribution = pd.crosstab(dataset.frame['label_name'], dataset.frame['split'])
        distribution['total'] = distribution.sum(axis=1)
        display(distribution)
        train_counts = distribution['train']
        print('Train imbalance ratio:', round(float(train_counts.max() / train_counts.min()), 3))
    finally:
        dataset = distribution = train_counts = None
        gc.collect()
print('All image paths, labels, official splits, and fingerprints passed validation.')
"""


def _update_training(source: str) -> str:
    block = (
        RUN_BEGIN
        + "if RUN_CONTINUAL_BASELINES or RUN_RANPAC_STABILITY or RUN_RANPAC_MECHANISM_ABLATION or RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT or RUN_RANPAC_CLASS_BALANCED_RIDGE or RUN_RANPAC_CLASS_BALANCED_CONFIRMATION or RUN_CONTINUAL_CONFIRMATION:\n"
        + "    from experiments.mm_hope.baseline_preflight import write_baseline_preflight\n"
        + "    from experiments.mm_hope.frozen_baseline_runner import (\n"
        + "        RANPAC_CLASS_BALANCED_RIDGE_CONFIRMATION_CANDIDATES,\n"
        + "        lock_frozen_baseline_development, lock_ranpac_class_balanced_ridge_development,\n"
        + "        run_frozen_baseline_confirmation, run_ranpac_class_balanced_ridge_confirmation,\n"
        + "        run_frozen_baseline_development, run_ranpac_class_balanced_ridge_development,\n"
        + "        run_ranpac_class_prior_adjustment_development,\n"
        + "        run_ranpac_mechanism_ablation_development, run_ranpac_stability_development,\n"
        + "        summarize_ranpac_class_balanced_ridge_confirmation,\n"
        + "    )\n"
        + "\n"
        + "    CONTINUAL_OUTPUT_ROOT = PROJECT_ROOT / 'analysis_outputs' / 'continual_baseline_s_frozen_v1'\n"
        + "    write_baseline_preflight(\n"
        + "        PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "        seed=2026, update_quantum=32, alignment_quantum=256, preconditioner_quantum=128,\n"
        + "    )\n"
        + "    if RUN_CONTINUAL_BASELINES:\n"
        + "        continual_results = run_frozen_baseline_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            methods=CONTINUAL_METHODS, seed=2026, device='cuda',\n"
        + "            feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS, local_files_only=False,\n"
        + "        )\n"
        + "        print('Continual validation-only runs:', len(continual_results))\n"
        + "    if RUN_RANPAC_STABILITY:\n"
        + "        ranpac_stability = run_ranpac_stability_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            seed=2026, device='cuda', feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS,\n"
        + "            local_files_only=False,\n"
        + "        )\n"
        + "        print('RanPAC stability runs:', len(ranpac_stability))\n"
        + "    if RUN_RANPAC_MECHANISM_ABLATION:\n"
        + "        ranpac_ablation = run_ranpac_mechanism_ablation_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            seed=2026, device='cuda', feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS,\n"
        + "            local_files_only=False,\n"
        + "        )\n"
        + "        print('RanPAC mechanism-ablation runs:', len(ranpac_ablation))\n"
        + "    if RUN_RANPAC_CLASS_PRIOR_ADJUSTMENT:\n"
        + "        ranpac_prior = run_ranpac_class_prior_adjustment_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            seed=2026, device='cuda', feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS,\n"
        + "            local_files_only=False,\n"
        + "        )\n"
        + "        print('RanPAC class-prior adjustment runs:', len(ranpac_prior))\n"
        + "    if RUN_RANPAC_CLASS_BALANCED_RIDGE:\n"
        + "        ranpac_balanced = run_ranpac_class_balanced_ridge_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "            candidate_ids=RANPAC_CLASS_BALANCED_CANDIDATES,\n"
        + "            seed=2026, device='cuda', feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS,\n"
        + "            local_files_only=False,\n"
        + "        )\n"
        + "        print('RanPAC class-balanced ridge runs:', len(ranpac_balanced))\n"
        + "    if RUN_RANPAC_CLASS_BALANCED_CONFIRMATION:\n"
        + "        assert not RUN_RANPAC_CLASS_BALANCED_RIDGE, 'Run the validation-only bridge control before test confirmation.'\n"
        + "        try:\n"
        + "            lock_ranpac_class_balanced_ridge_development(\n"
        + "                PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "            )\n"
        + "        except FileNotFoundError as error:\n"
        + "            if 'missing complete validation-only candidate required for confirmation' not in str(error):\n"
        + "                raise\n"
        + "            print('Locked development artifacts are absent in this runtime; rebuilding only beta=0 and beta=0.50 on validation.')\n"
        + "            bridge_results = run_ranpac_class_balanced_ridge_development(\n"
        + "                PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "                candidate_ids=RANPAC_CLASS_BALANCED_RIDGE_CONFIRMATION_CANDIDATES,\n"
        + "                seed=2026, device='cuda', feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "                num_workers=FEATURE_NUM_WORKERS, local_files_only=False,\n"
        + "            )\n"
        + "            print('Rebuilt validation-only lock artifacts:', len(bridge_results))\n"
        + "            lock_ranpac_class_balanced_ridge_development(\n"
        + "                PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "            )\n"
        + "        ranpac_balanced_confirmation = run_ranpac_class_balanced_ridge_confirmation(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "            seeds=RANPAC_CLASS_BALANCED_CONFIRMATION_SEEDS, device='cuda',\n"
        + "            feature_batch_size=FROZEN_FEATURE_BATCH_SIZE, num_workers=FEATURE_NUM_WORKERS,\n"
        + "            local_files_only=False, resume=True,\n"
        + "        )\n"
        + "        ranpac_balanced_summary = summarize_ranpac_class_balanced_ridge_confirmation(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=RANPAC_CLASS_BALANCED_DATASETS,\n"
        + "            bootstrap_replicates=BOOTSTRAP_REPLICATES,\n"
        + "        )\n"
        + "        print('RanPAC class-balanced confirmation runs:', len(ranpac_balanced_confirmation))\n"
        + "        print(json.dumps(ranpac_balanced_summary, indent=2))\n"
        + "    if RUN_CONTINUAL_CONFIRMATION:\n"
        + "        lock_frozen_baseline_development(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            methods=CONTINUAL_METHODS,\n"
        + "        )\n"
        + "        continual_confirmation = run_frozen_baseline_confirmation(\n"
        + "            PROJECT_ROOT, output_root=CONTINUAL_OUTPUT_ROOT, datasets=SELECTED_DATASETS,\n"
        + "            methods=CONTINUAL_METHODS, seeds=CONTINUAL_CONFIRMATION_SEEDS, device='cuda',\n"
        + "            feature_batch_size=FROZEN_FEATURE_BATCH_SIZE,\n"
        + "            num_workers=FEATURE_NUM_WORKERS, local_files_only=False,\n"
        + "        )\n"
        + "        print('Continual test-confirmation runs:', len(continual_confirmation))\n"
        + "    release_runtime_memory('cuda')\n"
        + RUN_END
    )
    start = source.find(RUN_BEGIN)
    if start >= 0:
        finish = source.find(RUN_END, start)
        if finish < 0:
            raise RuntimeError("missing continual development closing marker")
        source = source[:start] + source[finish + len(RUN_END) :]
    static_needle = "if RUN_PRESET == 'quick':\n"
    if "if RUN_PRESET == 'baseline_only':" not in source:
        if static_needle not in source:
            raise RuntimeError("notebook static training branch changed unexpectedly")
        source = source.replace(
            static_needle,
            "if RUN_PRESET == 'baseline_only':\n"
            "    print('Static G01--G06 training skipped: continual-baseline-only preset.')\n"
            "elif RUN_PRESET == 'quick':\n",
            1,
        )
    if "from experiments.mm_hope.gpu_study_trainer import release_runtime_memory" not in source:
        import_block = (
            "from experiments.mm_hope.gpu_study_runner import (\n"
            "    lock_development, run_confirmation, run_development, summarize_confirmation,\n"
            ")\n"
        )
        if import_block not in source:
            raise RuntimeError("notebook training imports changed unexpectedly")
        source = source.replace(
            import_block,
            import_block
            + "from experiments.mm_hope.gpu_study_trainer import release_runtime_memory\n"
            + "release_runtime_memory('cuda')\n",
            1,
        )
    replacements = {
        "models=('G01',), device='cuda', smoke=True, resume=True,\n": (
            "models=('G01',), device='cuda', smoke=True, resume=True,\n"
            "        num_workers=STATIC_NUM_WORKERS,\n"
        ),
        "models=static_models, device='cuda', smoke=False, resume=True,\n": (
            "models=static_models, device='cuda', smoke=False, resume=True,\n"
            "        num_workers=STATIC_NUM_WORKERS,\n"
        ),
        "models=SELECTED_MODELS, seeds=CONFIRMATION_SEEDS, device='cuda', resume=True,\n": (
            "models=SELECTED_MODELS, seeds=CONFIRMATION_SEEDS, device='cuda', resume=True,\n"
            "        num_workers=STATIC_NUM_WORKERS,\n"
        ),
    }
    for old, new in replacements.items():
        if old in source and new not in source:
            source = source.replace(old, new)
    needle = "print('Training stage complete:', RUN_PRESET)"
    if needle not in source:
        raise RuntimeError("notebook training cell changed unexpectedly")
    return source.replace(needle, block + needle)


def _update_metrics(source: str) -> str:
    start = source.find("result_roots = [OUTPUT_ROOT]\n")
    if start < 0:
        start = source.find("result_rows = []\n")
    end = source.find("summary_path = OUTPUT_ROOT / 'confirmation_summary.csv'\n")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("notebook metrics cell changed unexpectedly")
    table = """result_roots = [OUTPUT_ROOT]
if 'CONTINUAL_OUTPUT_ROOT' in globals() and CONTINUAL_OUTPUT_ROOT.is_dir():
    result_roots.append(CONTINUAL_OUTPUT_ROOT)
result_rows = []
for result_root in result_roots:
    for result_path in result_root.rglob('result.json'):
        result = json.loads(result_path.read_text(encoding='utf-8'))
        metrics = result.get('test_metrics') or result.get('validation_metrics', {})
        method_id = result.get('method_id')
        dataset_key = result.get('dataset')
        if dataset_key is None:
            dataset_key = next((part for part in result_path.parts if part in SELECTED_DATASETS), None)
        feature_peak = result.get('feature_extraction_peak_gpu_memory_bytes')
        if feature_peak is None and method_id:
            feature_path = result_path.parent.parent / 'feature_extraction.json'
            if feature_path.is_file():
                feature_peak = json.loads(feature_path.read_text(encoding='utf-8')).get(
                    'feature_extraction_peak_gpu_memory_bytes'
                )
        result_rows.append({
            'dataset': dataset_key,
            'model': result.get('config', {}).get('model_id') or method_id,
            'candidate': result.get('candidate_id'),
            'experiment_id': result.get('experiment_id'),
            'backbone': result.get('backbone') or ('G01_frozen_clip_vit_b32' if method_id else None),
            'protocol': result.get('protocol') or ('continual_s_frozen' if method_id else 'static_multimodal'),
            'seed': result.get('config', {}).get('seed'),
            'stage': result.get('stage'),
            'best_epoch': result.get('best_epoch'),
            'macro_f1': metrics.get('macro_f1'),
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'worst_class_f1': metrics.get('worst_class_f1'),
            'mcc': metrics.get('mcc'),
            'nll': metrics.get('nll'),
            'ece_15': metrics.get('ece_15'),
            'online_peak_vram_gib': (result.get('peak_gpu_memory_bytes') or result.get('online_peak_gpu_memory_bytes') or 0) / 2**30,
            'feature_extraction_peak_vram_gib': (feature_peak or 0) / 2**30,
        })
results = pd.DataFrame(result_rows)
if len(results):
    display(results.sort_values(['dataset', 'protocol', 'experiment_id', 'model', 'candidate', 'seed']).reset_index(drop=True))

"""
    return source[:start] + table + source[end:]


def _update_export(source: str) -> str:
    block = """# BEGIN MULTI-ROOT RESULT EXPORT
import zipfile
export_path = Path('/content/MM_HOPE_results.zip')
archive_roots = [OUTPUT_ROOT]
if 'CONTINUAL_OUTPUT_ROOT' in globals() and CONTINUAL_OUTPUT_ROOT.is_dir():
    archive_roots.append(CONTINUAL_OUTPUT_ROOT)
skip_suffixes = set() if INCLUDE_MODEL_CHECKPOINTS_IN_EXPORT else {'.pt', '.npy'}
with zipfile.ZipFile(export_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for archive_root in archive_roots:
        for path in archive_root.rglob('*'):
            if path.is_file() and path.suffix not in skip_suffixes:
                archive.write(path, Path(archive_root.name) / path.relative_to(archive_root))
print('Result archive:', export_path, '| MiB:', round(export_path.stat().st_size / 2**20, 2))
if DOWNLOAD_RESULTS_AT_END:
    from google.colab import files
    files.download(str(export_path))
# END MULTI-ROOT RESULT EXPORT
"""
    start = source.find(EXPORT_BEGIN)
    if start >= 0:
        end = source.find(EXPORT_END, start)
        if end < 0:
            raise RuntimeError("missing result export closing marker")
        return source[:start] + block + source[end + len(EXPORT_END) :]
    start = source.find("import zipfile\n")
    if start < 0:
        raise RuntimeError("notebook export cell changed unexpectedly")
    return source[:start] + block


def update_notebook() -> None:
    payload = _archive_payload()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    support = _cell_source(cells[3])
    support = _replace_marked_block(support, BEGIN, END, _patch_bundle_block(payload))
    _set_source(cells[3], support)

    _set_source(cells[1], _update_configuration(_cell_source(cells[1])))
    _set_source(cells[2], _update_system_setup(_cell_source(cells[2])))
    _set_source(cells[4], _safe_download_cell())
    _set_source(cells[5], _safe_install_cell())
    _set_source(cells[6], _safe_validation_cell())
    _set_source(cells[7], _update_training(_cell_source(cells[7])))
    metrics = _update_metrics(_cell_source(cells[8]))
    _set_source(cells[8], _update_export(metrics))

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    print(f"updated {NOTEBOOK.name}: {len(payload) / 2**10:.1f} KiB compressed patch")


if __name__ == "__main__":
    update_notebook()
