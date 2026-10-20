import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from homology import audit_homology_thresholds as audit


def training(identifier, sequence, source="one"):
    return {"id": identifier, "source": source, "sequence": sequence}


def heldout(identifier, sequence, split="test"):
    return {"id": identifier, "split": split, "sequence": sequence}


class HomologyTests(unittest.TestCase):
    def test_inclusive_thresholds_and_pair_vs_target_counts(self):
        members = [training(str(n), "A" * n + "C" * (20 - n)) for n in (16, 17, 18, 20)]
        rows, hits = audit.audit_overlaps(members, [heldout("query", "A" * 20)])
        self.assertEqual([rows[0][f"hits_{t}"] for t in (80, 85, 90)], [4, 3, 2])
        self.assertEqual([h["identity"] for h in hits if h["threshold"] == 0.9], [0.9, 1.0])
        summary = audit.summarize(members, rows)["splits"]["test"]
        self.assertEqual(summary["thresholds"]["80"], {"any_hit_targets": 1, "pair_count": 4})
        self.assertEqual(summary["exact_identity"], {"any_hit_targets": 1, "pair_count": 1})

    def test_coverage_is_bidirectional_and_inclusive(self):
        metrics = {"identity": 0.9, "heldout_coverage": 0.8, "train_coverage": 0.8}
        self.assertTrue(audit.qualifies(metrics, 0.9))
        for field in ("heldout_coverage", "train_coverage"):
            self.assertFalse(audit.qualifies(dict(metrics, **{field: 0.799999}), 0.8))
        metrics = audit.global_similarity("A" * 8, "A" * 10)
        self.assertEqual(metrics, {"identity": 0.8, "heldout_coverage": 1.0, "train_coverage": 0.8})
        self.assertTrue(audit.qualifies(metrics, 0.8))

    def test_missing_sequences_are_lower_bounds_not_zero_overlap(self):
        members = [training("known", "AAAA"), training("missing", None)]
        rows, hits = audit.audit_overlaps(members, [heldout("unmatched", "CCCC"), heldout("absent", None)])
        summary = audit.summarize(members, rows)
        self.assertEqual(summary["status"], "partial")
        self.assertTrue(summary["counts_are_lower_bounds"])
        self.assertIn("do not establish zero overlap", summary["interpretation"])
        self.assertEqual(summary["missing_training_members"][0]["id"], "missing")
        self.assertEqual(hits, [])
        self.assertTrue(all(not r["search_complete"] for r in rows))
        self.assertIsNone(rows[0]["hits_80"])
        self.assertEqual(rows[1]["hits_80"], 0)
        self.assertIsNone(rows[1]["best_qualifying_identity"])

    def test_source_preserving_membership_and_nonmember_queries(self):
        baseline = {"membership_evidence": {"kind": "toy"}, "training_members": [
            training("same", "ACGU", "one"), training("same", None, "two")]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            path.write_text("source,id,sequence\ntwo,same,GGGG\nthree,same,AAAA\none,new,CCCC\n", encoding="ascii")
            members, ignored = audit.resolve_training(baseline, [path])
        self.assertEqual([(m["source"], m["id"], m["sequence"]) for m in members],
                         [("one", "same", "ACGU"), ("two", "same", "GGGG")])
        self.assertEqual(len(ignored), 2)
        rows, _ = audit.audit_overlaps(members, [heldout("query", "GGGG")])
        self.assertEqual(rows[0]["best_qualifying_train_source"], "two")
        self.assertEqual(len(baseline["training_members"]), 2)

    def test_existing_new_and_duplicate_new_disagreements_fail(self):
        baseline = {"membership_evidence": {"kind": "toy"}, "training_members": [training("x", "ACGU")]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            path.write_text(json.dumps([training("x", "AAAA")]), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "query disagreement"):
                audit.resolve_training(baseline, [path])
            baseline["training_members"][0]["sequence"] = None
            path.write_text(json.dumps([training("x", "ACGU"), training("x", "AAAA")]), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "query disagreement"):
                audit.resolve_training(baseline, [path])

    def test_roots_first_record_insertions_and_conflict_validation(self):
        baseline = {"membership_evidence": {"kind": "toy"}, "training_members": [training("x", None)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x.a3m").write_text(">query\nACgg-G.Ut\n>not_query\nAAAA\n", encoding="ascii")
            (root / "nonmember.fasta").write_text(">extra\nCCCC\n", encoding="ascii")
            members, ignored = audit.resolve_training(baseline, root_specs=[f"one={root}"])
            self.assertEqual(members[0]["sequence"], "ACGU")
            self.assertEqual(len(members), 1)
            self.assertEqual(ignored, [])
            (root / "x.fasta").write_text(">query\nacgu\n", encoding="ascii")
            audit.resolve_training(baseline, root_specs=[f"one={root}"])
            (root / "x.fasta").write_text(">query\nAAAA\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "query disagreement"):
                audit.resolve_training(baseline, root_specs=[f"one={root}"])
        self.assertEqual(audit.normalize("acg-t.u"), "ACGUU")
        self.assertEqual(audit.normalize("ACgg-G.Ut", "a2m"), "ACGU")

    def test_length_pruning_keeps_boundary_and_does_not_claim_nearest(self):
        members = [training("boundary", "A" * 8), training("too_short", "A" * 7)]
        rows, _ = audit.audit_overlaps(members, [heldout("q", "A" * 10)])
        self.assertEqual(rows[0]["aligned_pairs"], 1)
        self.assertEqual(rows[0]["length_pruned_pairs"], 1)
        self.assertEqual(rows[0]["hits_80"], 1)
        self.assertEqual(rows[0]["best_qualifying_identity"], 0.8)
        rows, _ = audit.audit_overlaps(members[1:], [heldout("q", "A" * 10)])
        self.assertIsNone(rows[0]["best_qualifying_identity"])

    def test_tie_break_is_independent_of_member_order(self):
        members = [training("z", "ACGU", "b"), training("z", "ACGU", "a"), training("a", "ACGU", "a")]
        query = [heldout("q", "ACGU")]
        forward = audit.audit_overlaps(members, query)
        self.assertEqual(forward, audit.audit_overlaps(list(reversed(members)), query))
        self.assertEqual(forward[0][0]["best_qualifying_train_source"], "a")
        self.assertEqual(forward[0][0]["best_qualifying_train_id"], "a")

    def test_ambiguous_literal_equality_is_not_exact_identity(self):
        members = [training("x", "NNNN")]
        rows, hits = audit.audit_overlaps(members, [heldout("q", "NNNN")])
        self.assertEqual(rows[0]["literal_equal_pair_count"], 1)
        self.assertEqual(rows[0]["exact_identity_pair_count"], 0)
        self.assertEqual(hits, [])

    def test_process_pool_is_deterministic(self):
        members = [training("b", "ACGU"), training("a", "ACGU"), training("missing", None)]
        queries = [heldout("z", "ACGU"), heldout("b", "AAAA", "validation"), heldout("a", None)]
        self.assertEqual(audit.audit_overlaps(members, queries, workers=1),
                         audit.audit_overlaps(members, queries, workers=2))

    def test_complete_zero_hits_vs_missing_heldout_only(self):
        members = [training("x", "AAAA")]
        rows, _ = audit.audit_overlaps(members, [heldout("q", "CCCC")])
        summary = audit.summarize(members, rows)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["counts_are_lower_bounds"])
        self.assertTrue(rows[0]["search_complete"])
        rows, _ = audit.audit_overlaps(members, [heldout("q", None)])
        summary = audit.summarize(members, rows)
        self.assertEqual(summary["status"], "partial")
        self.assertTrue(summary["training_sequence_resolution_complete"])
        self.assertTrue(summary["splits"]["test"]["counts_are_lower_bounds"])
        self.assertIsNone(rows[0]["hits_80"])

    def test_membership_guards_and_unresolved_training(self):
        baseline = {"membership_evidence": {"kind": "toy"}, "training_members": [training("x", None)]}
        members, _ = audit.resolve_training(baseline)
        rows, _ = audit.audit_overlaps(members, [heldout("q", "ACGU")])
        self.assertEqual(audit.summarize(members, rows)["resolved_training_queries"], 0)
        self.assertFalse(rows[0]["search_complete"])
        for invalid in ({"training_members": baseline["training_members"]},
                        dict(baseline, training_members=[training("x", None)] * 2),
                        dict(baseline, training_members=[training("../x", None)]),
                        dict(baseline, training_members=[{"id": "x", "sequence": "ACGU"}])):
            with self.assertRaises(ValueError):
                audit.resolve_training(invalid)
        with self.assertRaisesRegex(ValueError, "known training SOURCE"):
            audit.resolve_training(baseline, root_specs=["unknown=."])
        with self.assertRaisesRegex(ValueError, "workers"):
            audit.audit_overlaps(members, [], workers=0)

    def test_math_matches_original_implementation(self):
        path = audit.ROOT / "RethinkingMSA_1/manuscript/recomb_borderline/audit_training.py"
        if not path.is_file() and Path(audit.__file__).parent.name == "homology":
            self.skipTest("Original private audit_training.py is not shipped in the portable release")
        spec = importlib.util.spec_from_file_location("original_training_audit", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        self.assertEqual(audit.SCORES, original.SCORES)
        for a, b in [("ACGU", "ACGU"), ("ACGUN", "ACGUU"), ("ACGUACGU", "AGU"),
                     ("A" * 20, "A" * 17 + "C" * 3), ("AGCUAG", "ACGUAGCU")]:
            expected = original.global_similarity(a, b)
            expected["heldout_coverage"] = expected.pop("test_coverage")
            self.assertEqual(audit.global_similarity(a, b), expected)

    def test_heldout_fastas_resolve_only_manifest_ids_without_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "benchmark.csv"
            manifest.write_text("target_id,split\nv,val\nt,test\nmissing,test\nother,train\n", encoding="ascii")
            val, test = root / "val.fasta", root / "test.fasta"
            val.write_text(">v description\nacg-t.u\n>extra\nAAAA\n", encoding="ascii")
            test.write_text(">t\nGG\nCC\n", encoding="ascii")
            rows = audit.load_heldout(manifest, root / "absent", fastas=[val, test])
            self.assertEqual([(r["id"], r["split"], r["sequence"]) for r in rows],
                             [("missing", "test", None), ("t", "test", "GGCC"), ("v", "validation", "ACGUU")])
            self.assertEqual(rows[1]["query_path"], str(test))
            self.assertEqual(rows[2]["query_path"], str(val))
            self.assertTrue(all(r["sequence"] is None for r in audit.load_heldout(manifest, root)))

    def test_heldout_explicit_missing_empty_and_invalid_fastas_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "benchmark.csv"
            manifest.write_text("id,split\nq,test\n", encoding="ascii")
            (root / "q").mkdir()
            (root / "q/q.fasta").write_text(">q\nACGU\n", encoding="ascii")
            missing = root / "missing.fasta"
            for path in (missing, root):
                with self.assertRaisesRegex(ValueError, "heldout FASTA is not a file"):
                    audit.load_heldout(manifest, root, [path])
            path = root / "queries.fasta"
            path.write_text("", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "no FASTA queries"):
                audit.load_heldout(manifest, root, [path])
            for text in (">q\nINVALID\n", ">q\n", ">../q\nACGU\n"):
                path.write_text(text, encoding="ascii")
                with self.assertRaises(ValueError):
                    audit.load_heldout(manifest, root, [path])

    def test_heldout_duplicate_and_conflicting_fasta_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "benchmark.csv"
            manifest.write_text("id,split\nq,test\n", encoding="ascii")
            first, second = root / "first.fasta", root / "second.fasta"
            for identifier in ("q", "nonmember"):
                first.write_text(f">{identifier}\nACGU\n", encoding="ascii")
                for sequence, message in (("acgt", "duplicate heldout FASTA query id"),
                                          ("AAAA", "heldout query disagreement")):
                    second.write_text(f">{identifier}\n{sequence}\n", encoding="ascii")
                    with self.assertRaisesRegex(ValueError, message):
                        audit.load_heldout(manifest, root, [first, second])
                    first.write_text(f">{identifier}\nACGU\n>{identifier}\n{sequence}\n", encoding="ascii")
                    with self.assertRaisesRegex(ValueError, message):
                        audit.load_heldout(manifest, root, [first])
                    first.write_text(f">{identifier}\nACGU\n", encoding="ascii")

    def test_heldout_fastas_must_agree_with_per_id_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "benchmark.csv"
            manifest.write_text("id,split\nq,test\n", encoding="ascii")
            (root / "q").mkdir()
            per_id = root / "q/q.fasta"
            per_id.write_text(">unrelated_header\nACGU\n>not_query\nAAAA\n", encoding="ascii")
            fasta = root / "heldout.fasta"
            fasta.write_text(">q description\nacg-t\n", encoding="ascii")
            row = audit.load_heldout(manifest, root, [fasta])[0]
            self.assertEqual(row["sequence"], "ACGU")
            self.assertEqual(row["query_path"], str(per_id))
            fasta.write_text(">q\nAAAA\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "heldout query disagreement"):
                audit.load_heldout(manifest, root, [fasta])

    def test_heldout_manifest_duplicate_ids_still_fail_with_fastas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, fasta = root / "benchmark.csv", root / "heldout.fasta"
            fasta.write_text(">q\nACGU\n", encoding="ascii")
            for split in ("val", "test"):
                manifest.write_text(f"id,split\nq,val\nq,{split}\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "duplicate heldout id"):
                    audit.load_heldout(manifest, root, [fasta])

    def test_portable_cli_defaults_point_to_public_metadata(self):
        args = audit.build_parser().parse_args(["--out_dir", "unused"])
        self.assertEqual(args.checkpoint_audit, audit.RELEASE_ROOT / "homology/primary8k_training_membership.json")
        self.assertEqual(args.benchmark_manifest, audit.RELEASE_ROOT / "homology/benchmark_manifest.csv")
        self.assertEqual(args.heldout_fasta, [])

    def test_heldout_fasta_cli_complete_and_input_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, manifest = root / "audit.json", root / "benchmark.csv"
            baseline.write_text(json.dumps({"membership_evidence": {"kind": "toy"},
                                           "training_members": [training("x", "ACGU")]}), encoding="ascii")
            manifest.write_text("id,split\nv,val\nt,test\n", encoding="ascii")
            val, test = root / "val.fasta", root / "test.fasta"
            val.write_text(">v\nACGU\n", encoding="ascii")
            test.write_text(">t\nAAAA\n", encoding="ascii")
            command = [sys.executable, str(Path(audit.__file__)), "--checkpoint_audit", str(baseline),
                       "--benchmark_manifest", str(manifest), "--heldout_input_dir", str(root / "absent"),
                       "--heldout_fasta", str(val), "--heldout_fasta", str(test),
                       "--out_dir", str(root / "out"), "--workers", "2"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["missing_heldout_queries"], [])
            self.assertEqual(summary["inputs"]["heldout_fastas"],
                             [{"path": str(p.resolve()), "sha256": audit.sha256(p)} for p in (val, test)])
            self.assertEqual(summary["splits"]["validation"]["thresholds"]["90"]["any_hit_targets"], 1)
            self.assertEqual(summary, json.loads((root / "out/summary.json").read_text()))
            val.unlink()
            failure = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("heldout FASTA is not a file", failure.stderr)

    def test_hpc_wrapper_runs_both_memberships_and_retains_partial_or_failed_runs(self):
        git = shutil.which("git")
        git_bash = Path(git).parent.parent / "bin/bash.exe" if git else Path("not-installed")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required to exercise the Slurm wrapper locally")
        script = audit.RELEASE_ROOT / "homology/run_audit_cpuhuge1t.sbatch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = root / "spool"
            spool.mkdir()
            script_copy = spool / "slurm_script"
            shutil.copyfile(script, script_copy)
            manifest = root / "benchmark.csv"
            manifest.write_text("id,split\nv,val\nt,test\n", encoding="ascii")
            fasta = root / "heldout.fasta"
            fasta.write_text(">v\nACGU\n>t\nAAAA\n", encoding="ascii")
            nufold, relaxed = root / "nufold", root / "relaxed"
            nufold.mkdir()
            relaxed.mkdir()
            (nufold / "x.fasta").write_text(">x\nACGU\n", encoding="ascii")
            (relaxed / "y.fasta").write_text(">y\nAAAA\n", encoding="ascii")
            primary, epoch = root / "primary.json", root / "epoch.json"
            epoch.write_text(json.dumps({"membership_evidence": "toy", "training_members": [
                {"source": "nufold", "id": "x"}, {"source": "nufolddb_relaxed_gap25", "id": "y"}]}), encoding="ascii")
            env = dict(os.environ, PYTHON=sys.executable, SLURM_SUBMIT_DIR=audit.RELEASE_ROOT.as_posix(),
                       SLURM_CPUS_PER_TASK="2", PRIMARY8K_MEMBERSHIP=primary.as_posix(),
                       EPOCH45_MEMBERSHIP=epoch.as_posix(), BENCHMARK_MANIFEST=manifest.as_posix(),
                       HELDOUT_VAL_FASTA=fasta.as_posix(), HELDOUT_TEST_FASTA="", HELDOUT_INPUT_DIR="",
                       QUERY_ROOT_NUFOLD=nufold.as_posix(), QUERY_ROOT_REPA_ROUND1=(root / "absent").as_posix(),
                       QUERY_ROOT_NUFOLDDB_ROUND3=(root / "absent").as_posix(),
                       QUERY_ROOT_NUFOLDDB_RELAXED_GAP25=relaxed.as_posix(), OUT_DIR=(root / "results").as_posix())
            env.pop("RELEASE_ROOT", None)
            env.pop("WORKERS", None)
            for scenario in ("complete", "partial", "error"):
                with self.subTest(scenario=scenario):
                    identifier = "missing" if scenario == "partial" else "x"
                    primary.write_text("not-json" if scenario == "error" else json.dumps({
                        "membership_evidence": "toy", "training_members": [{"source": "nufold", "id": identifier}]}), encoding="ascii")
                    env["SLURM_JOB_ID"] = scenario
                    if scenario == "error":
                        env["WORKERS"] = "1"
                        env["SLURM_SUBMIT_DIR"] = (audit.RELEASE_ROOT / "homology").as_posix()
                    result = subprocess.run([bash, script_copy.as_posix()], cwd=spool, env=env,
                                            capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0 if scenario == "complete" else 1,
                                     result.stdout + result.stderr)
                    self.assertIn("Workers: 1" if scenario == "error" else "Workers: 2", result.stdout)
                    outputs = root / "results" / f"job-{scenario}"
                    second = json.loads((outputs / "epoch45/summary.json").read_text())
                    self.assertEqual(second["status"], "complete")
                    self.assertTrue((outputs / "primary8k/audit.stderr.log").is_file())
                    if scenario != "error":
                        first = json.loads((outputs / "primary8k/summary.json").read_text())
                        self.assertEqual(first["status"], scenario)
                    else:
                        self.assertIn("ERROR", result.stderr)

    def test_hpc_wrapper_never_defaults_to_script_or_spool_directory(self):
        git = shutil.which("git")
        git_bash = Path(git).parent.parent / "bin/bash.exe" if git else Path("not-installed")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required to exercise the Slurm wrapper locally")
        env = dict(os.environ)
        env.pop("RELEASE_ROOT", None)
        env.pop("SLURM_SUBMIT_DIR", None)
        result = subprocess.run([bash, (audit.RELEASE_ROOT / "homology/run_audit_cpuhuge1t.sbatch").as_posix()],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("RELEASE_ROOT", result.stderr)

    def test_toy_cli_outputs_and_missing_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "audit.json"
            baseline.write_text(json.dumps({"membership_evidence": {"kind": "toy"},
                                           "training_members": [training("x", "ACGU"), training("missing", None)]}), encoding="ascii")
            manifest = root / "benchmark.csv"
            manifest.write_text("target_id,split\nq,validation\nabsent,test\n", encoding="ascii")
            (root / "q").mkdir()
            (root / "q/q.fasta").write_text(">query\nACGU\n", encoding="ascii")
            args = audit.build_parser().parse_args([
                "--checkpoint_audit", str(baseline), "--benchmark_manifest", str(manifest),
                "--heldout_input_dir", str(root), "--out_dir", str(root / "out")])
            summary = audit.run_audit(args)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["splits"]["validation"]["thresholds"]["90"]["any_hit_targets"], 1)
            self.assertEqual(len(summary["missing_heldout_queries"]), 1)
            disk = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary, disk)
            with (root / "out/threshold_hits.csv").open(newline="") as handle:
                hits = list(csv.DictReader(handle))
            self.assertEqual([float(r["threshold"]) for r in hits], [0.8, 0.85, 0.9])
            with (root / "out/missing_training_ids.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["id"], "missing")
            result = subprocess.run(
                [sys.executable, str(Path(audit.__file__)), "--checkpoint_audit", str(baseline),
                 "--benchmark_manifest", str(manifest), "--heldout_input_dir", str(root),
                 "--out_dir", str(root / "out_pool"), "--workers", "2"],
                capture_output=True, text=True, timeout=30, check=True)
            self.assertEqual(json.loads(result.stdout), summary)
            for filename in ("summary.json", "threshold_hits.csv", "heldout_best_qualifying.csv",
                             "missing_training_ids.csv", "resolved_training_queries.csv"):
                self.assertEqual((root / "out" / filename).read_bytes(),
                                 (root / "out_pool" / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
