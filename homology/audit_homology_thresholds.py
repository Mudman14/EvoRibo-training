"""Audit fixed checkpoint membership against validation/test RNA queries.

No checkpoint loading, membership discovery, or network access. Extra query CSV
or JSON manifests contain id/source/sequence records; nonmembers are ignored and
reported, never added to training. Repeat --query_root SOURCE=PATH to resolve
first-record FASTA/A2M/A3M queries. Lowercase insertions are removed for A2M/A3M
files (or manifest records with format=a2m/a3m), not ordinary FASTA sequences.
Repeat --heldout_fasta PATH to resolve benchmark IDs from multi-record FASTAs;
provided files must exist and agree with any per-ID heldout query files.
Only best qualifying hits are reported: no nearest-neighbor claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

from Bio import Align, SeqIO


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = (ROOT if Path(__file__).resolve().parent.name == "homology"
                else Path(__file__).resolve().parent / "EvoRibo-training")
DEFAULT_AUDIT = RELEASE_ROOT / "homology/primary8k_training_membership.json"
SCORES = {"match_score": 2.0, "mismatch_score": -1.0,
          "open_gap_score": -2.0, "extend_gap_score": -0.5}
THRESHOLDS = (0.8, 0.85, 0.9)
SUFFIXES = (".fasta", ".fa", ".fna", ".a2m", ".a3m")
PAIR_FIELDS = ("heldout_id", "split", "train_source", "train_id", "identity",
               "heldout_coverage", "train_coverage", "literal_sequence_equal")
TARGET_FIELDS = ("heldout_id", "split", "status", "search_complete",
                 "aligned_pairs", "length_pruned_pairs", "exact_identity_pair_count",
                 "literal_equal_pair_count", "hits_80", "hits_85", "hits_90",
                 "best_qualifying_train_source", "best_qualifying_train_id",
                 "best_qualifying_identity", "best_qualifying_heldout_coverage",
                 "best_qualifying_train_coverage")
_TRAINING = ()
_COMPLETE = False


def normalize(sequence, alignment_format=""):
    sequence = "".join(str(sequence).split())
    if alignment_format.lower().lstrip(".") in {"a2m", "a3m"}:
        sequence = "".join(c for c in sequence if not c.islower())
    sequence = sequence.upper().replace("T", "U").replace("-", "").replace(".", "")
    if not sequence or set(sequence) - set("ACGURYSWKMBDHVN"):
        raise ValueError("empty or invalid RNA query")
    return sequence


def global_similarity(test, train):
    """Exact scoring/first optimal alignment math from audit_training.py."""
    aligner = Align.PairwiseAligner(mode="global", **SCORES)
    alignment = aligner.align(test, train)[0]
    paired = matches = columns = 0
    coordinates = alignment.coordinates
    for index in range(coordinates.shape[1] - 1):
        a, b = coordinates[:, index]
        c, d = coordinates[:, index + 1]
        columns += max(c - a, d - b)
        if c > a and d > b:
            paired += c - a
            matches += sum(x == y and x in "ACGU" for x, y in zip(test[a:c], train[b:d]))
    return {"identity": float(matches / columns),
            "heldout_coverage": float(paired / len(test)),
            "train_coverage": float(paired / len(train))}


def qualifies(metrics, threshold):
    return (metrics["identity"] >= threshold and metrics["heldout_coverage"] >= 0.8
            and metrics["train_coverage"] >= 0.8)


def length_can_qualify(a, b):
    # matches <= min(lengths), alignment columns >= max(lengths). Strict <80%
    # is the only pruning criterion, so equality at the boundary is retained.
    return 5 * min(a, b) >= 4 * max(a, b)


def key(row):
    source, identifier = row.get("source"), row.get("id")
    if not isinstance(source, str) or not source or not isinstance(identifier, str) or not identifier:
        raise ValueError("training/query records require nonempty string source and id")
    if identifier in {".", ".."} or any(c in identifier for c in "/\\:"):
        raise ValueError(f"unsafe query id: {identifier!r}")
    return source, identifier


def first_query(path):
    with path.open(encoding="utf-8-sig") as handle:
        record = next(SeqIO.parse(handle, "fasta"), None)
    if record is None:
        raise ValueError(f"no FASTA query in {path}")
    return normalize(record.seq, path.suffix)


def query_candidates(root, identifier):
    for suffix in SUFFIXES:
        for path in (root / (identifier + suffix), root / identifier / (identifier + suffix)):
            if path.is_file():
                yield path


def read_query_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        data = list(csv.DictReader(handle)) if path.suffix.lower() == ".csv" else json.load(handle)
    if isinstance(data, dict):
        data = data.get("records")
    if not isinstance(data, list):
        raise ValueError(f"query manifest must contain a record list: {path}")
    return data


def resolve_training(audit, manifests=(), root_specs=()):
    if not audit.get("membership_evidence") or not audit.get("training_members"):
        raise ValueError("checkpoint audit must contain known training membership and membership_evidence")
    members = {}
    origins = {}

    def add_query(member_key, sequence, origin, alignment_format=""):
        value = normalize(sequence, alignment_format)
        previous = members[member_key]["sequence"]
        if previous is not None and previous != value:
            raise ValueError(f"query disagreement for {member_key}: {origins[member_key]} vs {origin}")
        members[member_key]["sequence"] = value
        origins[member_key].append(origin)

    for row in audit["training_members"]:
        member_key = key(row)
        if member_key in members:
            raise ValueError(f"duplicate training membership: {member_key}")
        members[member_key] = {"source": member_key[0], "id": member_key[1], "sequence": None}
        origins[member_key] = []
        if row.get("sequence"):
            add_query(member_key, row["sequence"], "checkpoint_audit")
    ignored = set()
    for path in manifests:
        for row in read_query_manifest(Path(path)):
            member_key = key(row)
            if not row.get("sequence"):
                raise ValueError(f"query manifest record has no sequence: {member_key}")
            if member_key not in members:
                ignored.add(member_key)
                continue
            add_query(member_key, row["sequence"], str(path), row.get("format", ""))
    roots = {}
    sources = {source for source, _ in members}
    for spec in root_specs:
        source, separator, directory = spec.partition("=")
        if not separator or source not in sources or not directory:
            raise ValueError("--query_root requires a known training SOURCE=PATH")
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"query root is not a directory: {root}")
        roots.setdefault(source, []).append(root)
    for member_key in sorted(members):
        source, identifier = member_key
        for root in roots.get(source, []):
            for path in query_candidates(root, identifier):
                add_query(member_key, first_query(path), str(path))
    rows = [dict(members[k], query_origins=origins[k]) for k in sorted(members)]
    ignored_rows = [{"source": s, "id": i} for s, i in sorted(ignored)]
    return rows, ignored_rows


def load_heldout(manifest, input_dir, fastas=()):
    queries, origins = {}, {}
    for fasta in fastas:
        path = Path(fasta)
        if not path.is_file():
            raise ValueError(f"heldout FASTA is not a file: {path}")
        count = 0
        with path.open(encoding="utf-8-sig") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                identifier = record.id
                key({"source": "benchmark", "id": identifier})
                sequence = normalize(record.seq)
                if identifier in queries:
                    if queries[identifier] != sequence:
                        raise ValueError(f"heldout query disagreement for {identifier}: {origins[identifier]} vs {path}")
                    raise ValueError(f"duplicate heldout FASTA query id: {identifier} ({origins[identifier]} vs {path})")
                queries[identifier], origins[identifier] = sequence, str(path)
                count += 1
        if not count:
            raise ValueError(f"no FASTA queries in {path}")
    rows, seen = [], set()
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            split = record.get("split", "").lower()
            if not split:
                raise ValueError("benchmark manifest requires an explicit split")
            if split == "val":
                split = "validation"
            if split not in {"validation", "test"}:
                continue
            identifier = record.get("target_id") or record.get("id")
            key({"source": "benchmark", "id": identifier})
            if identifier in seen:
                raise ValueError(f"duplicate heldout id (including cross-split): {identifier}")
            seen.add(identifier)
            path = Path(input_dir) / identifier / f"{identifier}.fasta"
            sequence = first_query(path) if path.is_file() else None
            query_path = str(path)
            if identifier in queries:
                if sequence is not None and sequence != queries[identifier]:
                    raise ValueError(f"heldout query disagreement for {identifier}: {path} vs {origins[identifier]}")
                if sequence is None:
                    sequence, query_path = queries[identifier], origins[identifier]
            rows.append({"id": identifier, "split": split, "sequence": sequence,
                         "query_path": query_path})
    if not rows:
        raise ValueError("benchmark contains no validation/test members")
    return sorted(rows, key=lambda row: (row["split"], row["id"]))


def compare_heldout(heldout, training, training_complete):
    result = dict.fromkeys(TARGET_FIELDS)
    result.update(heldout_id=heldout["id"], split=heldout["split"],
                  status="resolved" if heldout["sequence"] else "missing_query",
                  search_complete=training_complete and bool(heldout["sequence"]),
                  aligned_pairs=0, length_pruned_pairs=0)
    hits = []
    if not heldout["sequence"]:
        return result, hits
    result.update(exact_identity_pair_count=0, literal_equal_pair_count=0,
                  hits_80=0, hits_85=0, hits_90=0)
    best, best_rank = None, None
    for member in training:
        test, train = heldout["sequence"], member["sequence"]
        if not length_can_qualify(len(test), len(train)):
            result["length_pruned_pairs"] += 1
            continue
        metrics = global_similarity(test, train)
        result["aligned_pairs"] += 1
        literal_equal = test == train
        result["literal_equal_pair_count"] += int(literal_equal)
        result["exact_identity_pair_count"] += int(metrics["identity"] == 1.0)
        if not qualifies(metrics, 0.8):
            continue
        pair = {"heldout_id": heldout["id"], "split": heldout["split"],
                "train_source": member["source"], "train_id": member["id"],
                **metrics, "literal_sequence_equal": literal_equal}
        rank = (-metrics["identity"], -min(metrics["heldout_coverage"], metrics["train_coverage"]),
                member["source"], member["id"])
        if best_rank is None or rank < best_rank:
            best, best_rank = pair, rank
        for threshold in THRESHOLDS:
            if qualifies(metrics, threshold):
                result[f"hits_{round(100 * threshold)}"] += 1
                hits.append({"threshold": threshold, **pair})
    if best is not None:
        for field in ("train_source", "train_id", "identity", "heldout_coverage", "train_coverage"):
            result["best_qualifying_" + field] = best[field]
    return result, sorted(hits, key=lambda row: (row["threshold"], row["train_source"], row["train_id"]))


def _init_worker(training, complete):
    global _TRAINING, _COMPLETE
    _TRAINING, _COMPLETE = training, complete


def _worker(heldout):
    return compare_heldout(heldout, _TRAINING, _COMPLETE)


def audit_overlaps(training, heldout, workers=1):
    if workers < 1:
        raise ValueError("workers must be >= 1")
    resolved = sorted((r for r in training if r["sequence"]), key=key)
    complete = bool(training) and len(resolved) == len(training)
    heldout = sorted(heldout, key=lambda row: (row["split"], row["id"]))
    if workers == 1:
        results = [compare_heldout(row, resolved, complete) for row in heldout]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(resolved, complete)) as executor:
            results = list(executor.map(_worker, heldout))
    return [row for row, _ in results], [hit for _, hits in results for hit in hits]


def summarize(training, targets):
    missing = [{"source": row["source"], "id": row["id"], "reason": "query_not_resolved"}
               for row in training if not row["sequence"]]
    all_complete = bool(training) and not missing and all(r["search_complete"] for r in targets)
    splits = {}
    for split in ("validation", "test", "combined"):
        rows = [r for r in targets if split == "combined" or r["split"] == split]
        complete = bool(training) and not missing and all(r["search_complete"] for r in rows)
        splits[split] = {
            "heldout_targets": len(rows), "resolved_heldout_targets": sum(r["status"] == "resolved" for r in rows),
            "search_complete": complete, "counts_are_lower_bounds": not complete,
            "thresholds": {str(round(t * 100)): {
                "any_hit_targets": sum(bool(r[f"hits_{round(t * 100)}"]) for r in rows),
                "pair_count": sum(r[f"hits_{round(t * 100)}"] or 0 for r in rows)} for t in THRESHOLDS},
            "exact_identity": {"any_hit_targets": sum(bool(r["exact_identity_pair_count"]) for r in rows),
                               "pair_count": sum(r["exact_identity_pair_count"] or 0 for r in rows)},
            "literal_sequence_equality": {"any_hit_targets": sum(bool(r["literal_equal_pair_count"]) for r in rows),
                                          "pair_count": sum(r["literal_equal_pair_count"] or 0 for r in rows)}}
    return {"status": "complete" if all_complete else "partial",
            "counts_are_lower_bounds": not all_complete,
            "interpretation": ("Complete threshold search over fixed training membership."
                               if all_complete else "Partial search: observed overlap counts are lower bounds; zero observed hits do not establish zero overlap."),
            "training_members": len(training), "resolved_training_queries": len(training) - len(missing),
            "missing_training_queries": len(missing),
            "training_sequence_resolution_complete": bool(training) and not missing,
            "training_members_by_source": dict(sorted(Counter(r["source"] for r in training).items())),
            "resolved_training_queries_by_source": dict(sorted(Counter(r["source"] for r in training if r["sequence"]).items())),
            "missing_training_members": missing,
            "missing_heldout_queries": [{"id": r["heldout_id"], "split": r["split"]}
                                        for r in targets if r["status"] != "resolved"],
            "splits": splits}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args):
    audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8-sig"))
    training, ignored = resolve_training(audit, args.query_manifest, args.query_root)
    fastas = getattr(args, "heldout_fasta", ())
    heldout = load_heldout(args.benchmark_manifest, args.heldout_input_dir, fastas=fastas)
    targets, hits = audit_overlaps(training, heldout, args.workers)
    summary = summarize(training, targets)
    summary.update(
        membership_evidence=audit["membership_evidence"], ignored_nonmember_queries=ignored,
        checkpoint_sha256=audit.get("checkpoint_sha256"),
        inputs={"checkpoint_audit": str(args.checkpoint_audit.resolve()),
                "checkpoint_audit_sha256": sha256(args.checkpoint_audit),
                "benchmark_manifest": str(args.benchmark_manifest.resolve()),
                "benchmark_manifest_sha256": sha256(args.benchmark_manifest),
                "heldout_input_dir": str(args.heldout_input_dir.resolve()),
                "heldout_fastas": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in fastas],
                "query_manifests": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in args.query_manifest],
                "query_roots": args.query_root},
        alignment={"mode": "global", **SCORES,
                   "identity": "canonical A/C/G/U matches / all alignment columns including gaps",
                   "coverage": "paired residue columns / respective ungapped sequence length",
                   "thresholds_inclusive": list(THRESHOLDS), "bidirectional_coverage_minimum": 0.8,
                   "exact_identity": "identity == 1.0 (ambiguous bases are not matches)",
                   "literal_sequence_equality": "normalized string equality, including ambiguous bases",
                   "best_hit_scope": "qualifying hits at >=80% identity and >=80% bidirectional coverage only; null if none; not nearest neighbor",
                   "best_hit_tie_rule": "highest identity, highest minimum bidirectional coverage, lexicographically smallest (source, id)",
                   "alignment_tie_rule": "first Bio.Align.PairwiseAligner optimal alignment, as in audit_training.py",
                   "length_pruning": "skip only when min(lengths)/max(lengths) is strictly below 0.8"})
    import Bio
    summary["biopython_version"] = Bio.__version__
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "heldout_best_qualifying.csv", TARGET_FIELDS, targets)
    write_csv(args.out_dir / "threshold_hits.csv", ("threshold", *PAIR_FIELDS), hits)
    write_csv(args.out_dir / "missing_training_ids.csv", ("source", "id", "reason"), summary["missing_training_members"])
    write_csv(args.out_dir / "resolved_training_queries.csv", ("source", "id", "sequence", "query_origins"),
              [dict(row, query_origins=json.dumps(row["query_origins"])) for row in training if row["sequence"]])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--benchmark_manifest", type=Path,
                        default=RELEASE_ROOT / "homology/benchmark_manifest.csv")
    parser.add_argument("--heldout_input_dir", type=Path, default=RELEASE_ROOT / "homology/inputs")
    parser.add_argument("--heldout_fasta", type=Path, action="append", default=[], metavar="PATH",
                        help="Repeatable multi-record heldout FASTA; match benchmark IDs and reject duplicates/disagreements")
    parser.add_argument("--query_manifest", type=Path, action="append", default=[],
                        help="Repeatable CSV/JSON id/source/sequence records; source is mandatory")
    parser.add_argument("--query_root", action="append", default=[], metavar="SOURCE=PATH",
                        help="Check SOURCE queries at PATH/id.ext or PATH/id/id.ext; reject disagreements")
    parser.add_argument("--workers", type=int, default=1, help="Per-heldout process workers (default: serial)")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Write summary.json, threshold_hits.csv, heldout_best_qualifying.csv, missing_training_ids.csv, resolved_training_queries.csv")
    return parser


if __name__ == "__main__":
    print(json.dumps(run_audit(build_parser().parse_args()), indent=2))
