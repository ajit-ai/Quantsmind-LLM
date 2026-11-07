import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

SPLITS = ("train", "val", "test")
REQUIRED_SAMPLE_FIELDS = ("sample_id", "split", "image_path", "caption", "source")

# Fields that may appear when known. None/NULL means genuinely unknown —
# the loader never invents provenance or licensing values.
OPTIONAL_SAMPLE_FIELDS = (
    "source_url",
    "license",
    "attribution",
    "width",
    "height",
    # Real-data provenance (M1 real-run preparation):
    "checksum",  # hex sha256 of the image file bytes (as recorded)
    "acquisition_date",  # ISO-8601 date (YYYY-MM-DD) of acquisition
    "license_ref",  # URL / citation / registry id evidencing `license`
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def valid_sha256_hex(value: str) -> bool:
    """True when ``value`` looks like a 64-char lowercase hex sha256."""
    return bool(_SHA256_HEX.match(value))


def valid_iso_date(value: str) -> bool:
    """True when ``value`` is a real ISO-8601 calendar date (YYYY-MM-DD)."""
    if not _ISO_DATE.match(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    image_path: str
    caption: str
    source: str
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    width: int | None = None
    height: int | None = None
    preprocessing_version: str = "mm1_v2"
    checksum: str | None = None
    acquisition_date: str | None = None
    license_ref: str | None = None

    def resolved_path(self, manifest_dir: Path) -> Path:
        p = Path(self.image_path)
        if p.is_absolute():
            return p
        return manifest_dir / p


@dataclass
class Manifest:
    path: Path
    schema_version: int
    preprocessing_version: str
    dataset_name: str
    samples: list[SampleRecord] = field(default_factory=list)

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def for_split(self, split: str) -> list[SampleRecord]:
        if split not in SPLITS:
            raise ValueError(f"unknown split requested: {split!r} (valid: {SPLITS})")
        return [s for s in self.samples if s.split == split]


def load_manifest(manifest_path: str | Path) -> Manifest:
    """Load and validate a provenance-aware MANIFEST.json.

    Never invents provenance: any of the optional provenance fields that is
    absent or null stays None. Raises ValueError on a malformed manifest.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")

    if "schema_version" not in raw:
        raise ValueError("malformed manifest: missing 'schema_version'")
    if not isinstance(raw["schema_version"], int):
        raise ValueError("malformed manifest: 'schema_version' must be an integer")
    if "samples" not in raw or not isinstance(raw["samples"], list):
        raise ValueError("malformed manifest: 'samples' must be a list")

    dataset = raw.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("malformed manifest: 'dataset' must be an object")

    samples = []
    for i, s in enumerate(raw["samples"]):
        if not isinstance(s, dict):
            raise ValueError(f"malformed manifest: sample #{i} is not an object")
        missing = [f for f in REQUIRED_SAMPLE_FIELDS if f not in s]
        if missing:
            raise ValueError(f"malformed manifest: sample #{i} missing required field(s): {missing}")
        if not isinstance(s["sample_id"], str) or not s["sample_id"]:
            raise ValueError(f"malformed manifest: sample #{i} 'sample_id' must be a non-empty string")
        if s["split"] not in SPLITS:
            raise ValueError(f"malformed manifest: sample #{i} invalid split {s['split']!r}")
        if not isinstance(s["image_path"], str) or not s["image_path"]:
            raise ValueError(f"malformed manifest: sample #{i} 'image_path' must be a non-empty string")
        if not isinstance(s["caption"], str) or not s["caption"]:
            raise ValueError(f"malformed manifest: sample #{i} 'caption' must be a non-empty string")
        if not isinstance(s["source"], str) or not s["source"]:
            raise ValueError(f"malformed manifest: sample #{i} 'source' must be a non-empty string")
        width = s.get("width")
        height = s.get("height")
        if width is not None and (not isinstance(width, int) or width <= 0):
            raise ValueError(f"malformed manifest: sample #{i} 'width' must be a positive integer")
        if height is not None and (not isinstance(height, int) or height <= 0):
            raise ValueError(f"malformed manifest: sample #{i} 'height' must be a positive integer")

        checksum = s.get("checksum")
        if checksum is not None:
            if not isinstance(checksum, str) or not valid_sha256_hex(checksum):
                raise ValueError(
                    f"malformed manifest: sample #{i} 'checksum' must be a 64-char lowercase hex sha256"
                )
            checksum = checksum.lower()

        acquisition_date = s.get("acquisition_date")
        if acquisition_date is not None:
            if not isinstance(acquisition_date, str) or not valid_iso_date(acquisition_date):
                raise ValueError(
                    f"malformed manifest: sample #{i} 'acquisition_date' must be an ISO-8601 "
                    "date (YYYY-MM-DD)"
                )

        license_ref = s.get("license_ref")
        if license_ref is not None and (not isinstance(license_ref, str) or not license_ref):
            raise ValueError(
                f"malformed manifest: sample #{i} 'license_ref' must be a non-empty string"
            )

        samples.append(
            SampleRecord(
                sample_id=s["sample_id"],
                split=s["split"],
                image_path=s["image_path"],
                caption=s["caption"],
                source=s["source"],
                source_url=s.get("source_url"),
                license=s.get("license"),
                attribution=s.get("attribution"),
                width=width,
                height=height,
                preprocessing_version=s.get("preprocessing_version", "mm1_v2"),
                checksum=checksum,
                acquisition_date=acquisition_date,
                license_ref=license_ref,
            )
        )

    return Manifest(
        path=path,
        schema_version=raw["schema_version"],
        preprocessing_version=raw.get("preprocessing_version", "mm1_v2"),
        dataset_name=str(dataset.get("name", "")),
        samples=samples,
    )