"""
Secure dataset management API.
"""

import io
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from server.core import PROJECT_ROOT, ServiceManager, CIFAR10_CLASSES

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

CUSTOM_ROOT = PROJECT_ROOT / "data" / "custom_uploads"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. ()-]{0,119}$")


def _safe_dataset_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid dataset ID.")
    return value


def _safe_component(value: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid path component.")
    if ":" in value or "\\" in value or "/" in value:
        raise HTTPException(status_code=400, detail="Invalid path component.")
    return value


def _contained(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (ValueError, RuntimeError):
        return False


@router.get("/preview/{dataset_id}/{class_name}/{filename}")
def preview_image(dataset_id: str, class_name: str, filename: str):
    dataset_id = _safe_dataset_id(dataset_id)
    class_name = _safe_component(class_name)
    filename = _safe_component(filename)

    dataset_root = (CUSTOM_ROOT / dataset_id).resolve()
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    image_path = (dataset_root / class_name / filename).resolve()
    if not _contained(dataset_root, image_path):
        raise HTTPException(status_code=400, detail="Invalid image path.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    return FileResponse(str(image_path), filename=image_path.name)


@router.get("")
@router.get("/")
def list_datasets():
    datasets = [
        {
            "id": "cifar10",
            "name": "CIFAR-10",
            "status": "ready" if (PROJECT_ROOT / "data/cifar10").exists() else "not_found",
            "classes_count": 10,
            "classes": CIFAR10_CLASSES,
            "total_samples": 60000,
            "train_samples": 50000,
            "test_samples": 10000,
            "dimensions": "32x32",
            "description": "CIFAR-10 benchmark dataset.",
        }
    ]
    CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in sorted(CUSTOM_ROOT.iterdir(), key=lambda p: p.name):
        if not folder.is_dir() or not SAFE_ID.fullmatch(folder.name):
            continue
        manifest = folder / "manifest.json"
        name = folder.name
        if manifest.exists():
            try:
                import json
                with manifest.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                name = metadata.get("name", folder.name)
            except Exception:
                metadata = {}
        else:
            metadata = {}
        datasets.append({
            "id": folder.name,
            "name": name,
            "status": "ready",
            "classes_count": int(metadata.get("class_count", 0)),
            "total_samples": int(metadata.get("total_images", 0)),
            "dimensions": metadata.get("dimensions"),
            "description": "User-uploaded class-folder dataset.",
        })
    return {"success": True, "datasets": datasets}


@router.get("/inspect/{dataset_id}")
def inspect_dataset(dataset_id: str):
    if dataset_id == "cifar10":
        cifar_path = PROJECT_ROOT / "data/cifar10"
        if not cifar_path.exists():
            raise HTTPException(status_code=404, detail="CIFAR-10 directory not found.")
        manager = ServiceManager.get_instance()
        return {
            "success": True,
            "id": "cifar10",
            "name": "CIFAR-10",
            "total_images": 10000,
            "total_test_images": 10000,
            "classes": CIFAR10_CLASSES,
            "class_distribution": {name: 1000 for name in CIFAR10_CLASSES},
            "image_dimensions": [32, 32],
            "eval_input_size": [224, 224],
            "corrupted_images": 0,
            "recommended_splits": {"train": 80, "validation": 10, "test": 10},
            "samples_preview": manager.sample_cache[:10],
        }

    dataset_id = _safe_dataset_id(dataset_id)
    dataset_root = (CUSTOM_ROOT / dataset_id).resolve()
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Custom dataset not found.")

    class_counts: Dict[str, int] = {}
    preview: List[str] = []
    dimensions: Dict[str, List[int]] = {}
    corrupted = 0
    total = 0
    for class_folder in sorted(dataset_root.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        count = 0
        for image_path in sorted(class_folder.rglob("*")):
            if not image_path.is_file() or not _contained(dataset_root, image_path):
                continue
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    dims = [int(image.width), int(image.height)]
                dimensions.setdefault(class_name, dims)
            except (UnidentifiedImageError, OSError, ValueError):
                corrupted += 1
                continue
            count += 1
            total += 1
            if len(preview) < 12:
                preview.append(
                    f"/api/datasets/preview/{dataset_id}/{class_name}/{image_path.name}"
                )
        if count:
            class_counts[class_name] = count

    manifest = dataset_root / "manifest.json"
    name = dataset_id
    if manifest.exists():
        try:
            import json
            name = json.loads(manifest.read_text(encoding="utf-8")).get("name", name)
        except Exception:
            pass

    return {
        "success": True,
        "id": dataset_id,
        "name": name,
        "total_images": total,
        "classes": list(class_counts),
        "class_count": len(class_counts),
        "class_distribution": class_counts,
        "image_dimensions": dimensions,
        "corrupted_images": corrupted,
        "recommended_splits": {"train": 80, "validation": 20},
        "samples_preview": preview,
    }


@router.post("/upload")
def upload_custom_dataset(archive: UploadFile = File(...)):
    filename = archive.filename or "dataset.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP archives are supported.")

    contents = archive.file.read(MAX_ARCHIVE_BYTES + 1)
    if len(contents) > MAX_ARCHIVE_BYTES:
        raise HTTPException(status_code=400, detail="Archive exceeds 200 MB.")

    dataset_id = "ds_" + uuid.uuid4().hex
    dataset_root = (CUSTOM_ROOT / dataset_id).resolve()
    dataset_root.mkdir(parents=True, exist_ok=False)

    class_counts: Dict[str, int] = {}
    corrupted = 0
    total = 0
    dimension_set = set()

    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zf:
            for member in zf.infolist():
                raw_name = member.filename
                if member.is_dir():
                    continue
                if "\x00" in raw_name or raw_name.startswith(("/", "\\")):
                    raise HTTPException(status_code=400, detail="Unsafe absolute ZIP path.")
                if re.match(r"^[A-Za-z]:", raw_name):
                    raise HTTPException(status_code=400, detail="Unsafe drive-qualified ZIP path.")

                member_path = Path(raw_name.replace("\\", "/"))
                if ".." in member_path.parts or len(member_path.parts) < 2:
                    raise HTTPException(status_code=400, detail="ZIP entries must be inside class folders.")

                class_name = _safe_component(member_path.parts[-2])
                filename_component = _safe_component(member_path.name)
                target = (dataset_root / class_name / filename_component).resolve()
                if not _contained(dataset_root, target):
                    raise HTTPException(status_code=400, detail="ZIP Slip path rejected.")
                if member.external_attr & 0xA0000000 == 0xA0000000:
                    raise HTTPException(status_code=400, detail="Symlinks are not allowed.")
                if target.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as source:
                    data = source.read(MAX_ARCHIVE_BYTES + 1)
                if len(data) > MAX_ARCHIVE_BYTES:
                    raise HTTPException(status_code=400, detail="An extracted image exceeds the archive limit.")
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(data)) as image:
                        image_rgb = image.convert("RGB")
                        dimension_set.add((image.width, image.height))
                except (UnidentifiedImageError, OSError, ValueError):
                    corrupted += 1
                    continue
                target.write_bytes(data)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total += 1

        if total == 0:
            shutil.rmtree(dataset_root)
            raise HTTPException(status_code=400, detail="No valid class-folder images were found.")

        import json
        (dataset_root / "manifest.json").write_text(json.dumps({
            "name": Path(filename).stem,
            "class_count": len(class_counts),
            "total_images": total,
            "corrupted_images": corrupted,
            "dimensions": sorted(dimension_set),
        }, indent=2), encoding="utf-8")

        ServiceManager.get_instance().log_activity(
            "Dataset Uploaded",
            f"{filename}: {total} images across {len(class_counts)} classes.",
            "success",
        )
        return {
            "success": True,
            "dataset_id": dataset_id,
            "name": Path(filename).stem,
            "total_images": total,
            "classes_detected": list(class_counts),
            "class_count": len(class_counts),
            "class_distribution": class_counts,
            "corrupted_images": corrupted,
            "status": "VALIDATED",
        }
    except HTTPException:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        raise
    except Exception as exc:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        raise HTTPException(status_code=400, detail=f"Failed to process ZIP: {exc}")
