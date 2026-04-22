from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_QID_RE = re.compile(r"Q\d+")


@dataclass(frozen=True, slots=True)
class CeaGroundTruth:
    table_id: str
    row_id: int
    col_id: int
    gold_qids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cea_ground_truth_path(dataset_root: Path, dataset_id: str) -> Path:
    dataset_dir = dataset_root / dataset_id
    candidates = [
        dataset_dir / "gt" / "cea_gt.csv",
        dataset_dir / "gt" / "cea.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"CEA ground truth file not found. Checked: {joined}")


def _extract_qids(raw_value: str) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for qid in _QID_RE.findall(raw_value):
        if qid in seen:
            continue
        seen.add(qid)
        output.append(qid)
    return tuple(output)


def load_cea_ground_truth(dataset_root: Path, dataset_id: str) -> list[CeaGroundTruth]:
    path = cea_ground_truth_path(dataset_root, dataset_id)
    rows: list[CeaGroundTruth] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            gold_qids = _extract_qids(row[3])
            if not gold_qids:
                continue
            rows.append(
                CeaGroundTruth(
                    table_id=row[0],
                    row_id=int(row[1]),
                    col_id=int(row[2]),
                    gold_qids=gold_qids,
                )
            )
    return rows
