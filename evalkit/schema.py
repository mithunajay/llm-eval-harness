"""Versioned datasets with a strict schema.
A dataset is a JSONL file plus a sidecar meta file holding its version and sha256.
If the file changes without a re-freeze, loading fails loudly instead of silently
changing what an eval measures."""
import hashlib, json
from dataclasses import dataclass

TYPES = {"string": str, "integer": int, "array": list, "object": dict, "number": (int, float)}


class DatasetError(Exception):
    pass


def _type_ok(value, type_name):
    if type_name == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, TYPES[type_name])


def _validate(obj, fields, path, errors):
    if not isinstance(obj, dict):
        errors.append(f"{path or 'case'}: expected object")
        return
    for name, spec in fields.items():
        if isinstance(spec, str):
            spec = {"type": spec, "required": True}
        where = f"{path}.{name}" if path else name
        if name not in obj:
            if spec.get("required", True):
                errors.append(f"{where}: missing")
            continue
        if not _type_ok(obj[name], spec["type"]):
            errors.append(f"{where}: expected {spec['type']}")
            continue
        if "enum" in spec and obj[name] not in spec["enum"]:
            errors.append(f"{where}: must be one of {spec['enum']}")
        if spec["type"] == "object" and "fields" in spec:
            _validate(obj[name], spec["fields"], where, errors)
    for extra in sorted(set(obj) - set(fields)):
        errors.append(f"{path + '.' if path else ''}{extra}: unexpected field")


def validate_case(case, schema):
    errors = []
    _validate(case, schema["fields"], "", errors)
    return errors


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass
class Dataset:
    name: str
    version: str
    sha256: str
    cases: list


def meta_path_for(dataset_path):
    return dataset_path.rsplit(".", 1)[0] + ".meta.json"


def freeze(dataset_path, schema_path, name, version):
    """Validate the dataset, then record its version and hash in the meta file."""
    ds = load_dataset(dataset_path, schema_path, check_meta=False)
    meta = {"name": name, "version": version, "sha256": ds.sha256,
            "schema_version": json.load(open(schema_path))["schema_version"], "cases": len(ds.cases)}
    with open(meta_path_for(dataset_path), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_dataset(dataset_path, schema_path, check_meta=True):
    schema = json.load(open(schema_path))
    errors, cases, seen = [], [], set()
    with open(dataset_path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: invalid JSON ({e.msg})")
                continue
            for err in validate_case(case, schema):
                errors.append(f"line {n}: {err}")
            cid = case.get("id") if isinstance(case, dict) else None
            if cid in seen:
                errors.append(f"line {n}: duplicate id {cid!r}")
            seen.add(cid)
            cases.append(case)
    if errors:
        raise DatasetError("dataset failed validation:\n  " + "\n  ".join(errors))
    sha = file_sha256(dataset_path)
    name, version = dataset_path, "unfrozen"
    if check_meta:
        try:
            meta = json.load(open(meta_path_for(dataset_path)))
        except FileNotFoundError:
            raise DatasetError("no meta file: run `python -m evalkit freeze` first")
        if meta["sha256"] != sha:
            raise DatasetError(
                "dataset content changed since it was frozen (hash mismatch). "
                "Bump the version and re-freeze so baselines are not compared across different data.")
        name, version = meta["name"], meta["version"]
    return Dataset(name=name, version=version, sha256=sha, cases=cases)
