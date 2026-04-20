from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CeaTarget:
    table_id: str
    row_id: int
    col_id: int


@dataclass(frozen=True, slots=True)
class CtaTarget:
    table_id: str
    col_id: int


@dataclass(frozen=True, slots=True)
class CellContext:
    dataset_id: str
    table_id: str
    row_id: int
    col_id: int
    header: list[str]
    mention: str
    column_name: str
    row_values: list[str]
    other_row_values: list[str]
    sampled_column_values: list[str]
    mention_context: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def list_dataset_ids(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def resolve_dataset_dir(dataset_root: Path, dataset_id: str) -> Path:
    path = dataset_root / dataset_id
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of these files exist: {joined}")


def table_path(dataset_root: Path, dataset_id: str, table_id: str) -> Path:
    return resolve_dataset_dir(dataset_root, dataset_id) / "tables" / f"{table_id}.csv"


def cea_target_path(dataset_root: Path, dataset_id: str) -> Path:
    dataset_dir = resolve_dataset_dir(dataset_root, dataset_id)
    return _first_existing(
        [
            dataset_dir / "target" / "cea_target.csv",
            dataset_dir / "targets" / "CEA_2T_WD_Targets.csv",
        ]
    )


def cta_target_path(dataset_root: Path, dataset_id: str) -> Path:
    dataset_dir = resolve_dataset_dir(dataset_root, dataset_id)
    return _first_existing(
        [
            dataset_dir / "target" / "cta_target.csv",
            dataset_dir / "targets" / "CTA_2T_WD_targets.csv",
        ]
    )


def load_table(dataset_root: Path, dataset_id: str, table_id: str) -> list[list[str]]:
    path = table_path(dataset_root, dataset_id, table_id)
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def iter_cea_targets(dataset_root: Path, dataset_id: str) -> list[CeaTarget]:
    path = cea_target_path(dataset_root, dataset_id)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return [CeaTarget(table_id=row[0], row_id=int(row[1]), col_id=int(row[2])) for row in reader if len(row) >= 3]


def iter_cta_targets(dataset_root: Path, dataset_id: str) -> list[CtaTarget]:
    path = cta_target_path(dataset_root, dataset_id)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return [CtaTarget(table_id=row[0], col_id=int(row[1])) for row in reader if len(row) >= 2]


def build_cell_context(
    dataset_root: Path,
    dataset_id: str,
    table_id: str,
    *,
    row_id: int,
    col_id: int,
    sample_column_values: int = 5,
) -> CellContext:
    rows = load_table(dataset_root, dataset_id, table_id)
    if not rows:
        raise ValueError(f"Table is empty: {table_id}")
    if row_id <= 0:
        raise ValueError("row_id must point to a data row and be >= 1")
    if col_id < 0:
        raise ValueError("col_id must be >= 0")
    if row_id >= len(rows):
        raise IndexError(f"row_id {row_id} is out of range for table {table_id}")

    header = rows[0]
    if col_id >= len(header):
        raise IndexError(f"col_id {col_id} is out of range for table {table_id}")

    row_values = rows[row_id]
    mention = row_values[col_id].strip() if col_id < len(row_values) else ""
    column_name = header[col_id].strip() if col_id < len(header) else ""
    other_row_values = [
        value.strip()
        for index, value in enumerate(row_values)
        if index != col_id and value.strip()
    ]

    sampled_values: list[str] = []
    seen: set[str] = set()
    for raw_row in rows[1:]:
        if col_id >= len(raw_row):
            continue
        value = raw_row[col_id].strip()
        if not value or value == mention or value in seen:
            continue
        seen.add(value)
        sampled_values.append(value)
        if len(sampled_values) >= sample_column_values:
            break

    mention_context: list[str] = []
    for value in [column_name, *other_row_values, *sampled_values]:
        if value and value not in mention_context:
            mention_context.append(value)

    return CellContext(
        dataset_id=dataset_id,
        table_id=table_id,
        row_id=row_id,
        col_id=col_id,
        header=header,
        mention=mention,
        column_name=column_name,
        row_values=row_values,
        other_row_values=other_row_values,
        sampled_column_values=sampled_values,
        mention_context=mention_context,
    )


def default_lookup_payload(context: CellContext, *, top_k: int = 10) -> dict[str, object]:
    return {
        "mention": context.mention,
        "mention_context": context.mention_context,
        "coarse_hints": [],
        "fine_hints": [],
        "crosslink_hints": [],
        "top_k": int(top_k),
        "use_cache": False,
    }
