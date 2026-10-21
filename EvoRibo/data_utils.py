"""Small data utilities for EvoRibo experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

NUC_TO_IDX = {
    "A": 0,
    "U": 1,
    "C": 2,
    "G": 3,
    "-": 4,
    "N": 5,
}
IDX_TO_NUC = {v: k for k, v in NUC_TO_IDX.items()}


def _clean_rna_sequence(seq: str, keep_gaps: bool = False) -> str:
    clean: list[str] = []
    for ch in seq:
        if ch.islower():
            continue
        if ch == "-" and not keep_gaps:
            continue
        c = ch.upper()
        if c == "T":
            c = "U"
        allowed = set(NUC_TO_IDX) if keep_gaps else {"A", "U", "C", "G", "N"}
        if c in allowed:
            clean.append(c)
    return "".join(clean)


def _rows_are_dense_square(rows: list[list[float]]) -> bool:
    if not rows:
        return False
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        return False
    width = widths.pop()
    if width != len(rows):
        return False
    if width != 3:
        return True

    # A 3-column sparse file can coincidentally have 3 rows. Treat it as
    # sparse if the first two columns look like valid 1-indexed coordinates.
    coord_like = True
    for row in rows:
        i, j = row[0], row[1]
        if not (float(i).is_integer() and float(j).is_integer()):
            coord_like = False
            break
        if not (1 <= int(i) <= width and 1 <= int(j) <= width):
            coord_like = False
            break
    return not coord_like


def parse_prob_file(
    prob_path: str | Path,
    expected_length: int | None = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
) -> torch.Tensor:
    """Parse dense or sparse base-pair probabilities into a [L, L] tensor.

    Supported formats:
    - dense square matrix with whitespace-separated probabilities
    - three-column sparse rows: ``i j probability`` with 1-indexed positions
    """
    prob_path = Path(prob_path)
    rows: list[list[float]] = []

    with prob_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue

    if _rows_are_dense_square(rows):
        arr = np.asarray(rows, dtype=np.float32)
        mat = torch.from_numpy(arr.copy())
    else:
        pairs: list[tuple[int, int, float]] = []
        max_idx = 0
        for row in rows:
            if len(row) < 3:
                continue
            i, j, p = int(row[0]), int(row[1]), float(row[2])
            pairs.append((i, j, p))
            max_idx = max(max_idx, i, j)
        length = expected_length if expected_length is not None else max_idx
        if max_idx > length:
            raise ValueError(
                f"{prob_path} contains index {max_idx}, exceeding expected length {length}"
            )
        mat = torch.zeros(length, length, dtype=torch.float32)
        for i, j, p in pairs:
            mat[i - 1, j - 1] = p
            mat[j - 1, i - 1] = p

    if expected_length is not None and mat.shape != (expected_length, expected_length):
        raise ValueError(
            f"{prob_path} shape mismatch: expected {(expected_length, expected_length)}, "
            f"got {tuple(mat.shape)}"
        )
    if symmetrize:
        mat = torch.maximum(mat, mat.transpose(0, 1))
    if zero_diagonal and mat.numel() > 0:
        mat = mat.clone()
        mat.fill_diagonal_(0.0)
    return mat


def extract_query_seq_from_a3m(a3m_path: str | Path, keep_gaps: bool = False) -> str:
    """Extract the first A3M sequence, removing lowercase insertions."""
    a3m_path = Path(a3m_path)
    seq_lines: list[str] = []
    seen_header = False

    with a3m_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8", errors="ignore").replace("\r", "").rstrip("\n")
            if line.startswith(">"):
                if seen_header:
                    break
                seen_header = True
                continue
            if seen_header:
                seq_lines.append(line)

    return _clean_rna_sequence("".join(seq_lines), keep_gaps=keep_gaps)


def read_fasta_sequence(fasta_path: str | Path) -> str:
    """Read the first FASTA record as an ungapped RNA sequence."""
    fasta_path = Path(fasta_path)
    seq: list[str] = []
    seen_header = False
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header:
                    break
                seen_header = True
                continue
            if seen_header:
                seq.append(line)

    return _clean_rna_sequence("".join(seq), keep_gaps=False)


def tokenize_sequence(seq: str) -> torch.Tensor:
    """Tokenize an RNA sequence using A/U/C/G/gap/N ids."""
    tokens = []
    for ch in seq:
        c = ch.upper()
        if c == "T":
            c = "U"
        tokens.append(NUC_TO_IDX.get(c, NUC_TO_IDX["N"]))
    return torch.tensor(tokens, dtype=torch.long)


def dotbracket_to_bpp(dotbracket: str) -> torch.Tensor:
    """Convert dot-bracket style secondary structure into a binary BPP matrix."""
    open_to_close = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
    }
    close_to_open = {v: k for k, v in open_to_close.items()}
    stacks = {op: [] for op in open_to_close}

    L = len(dotbracket)
    mat = torch.zeros(L, L, dtype=torch.float32)

    for idx, ch in enumerate(dotbracket):
        if ch in open_to_close:
            stacks[ch].append(idx)
        elif ch in close_to_open:
            op = close_to_open[ch]
            if stacks[op]:
                j = stacks[op].pop()
                mat[idx, j] = 1.0
                mat[j, idx] = 1.0

    return mat
