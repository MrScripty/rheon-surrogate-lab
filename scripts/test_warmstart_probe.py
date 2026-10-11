import copy
import unittest
import probe_warmstart as probe


class ProbeProvenanceContracts(unittest.TestCase):
    def test_refuses_stale_dirty_unqualified_or_changed_prediction_reports(self):
        report = {
            "tool": {"lab_commit": "lab", "lab_dirty": "false", "patch_commit": "patch",
                     "patch_dirty": "false", "rheon_base": probe.BASE},
            "status": "complete", "warm": True,
            "cases": [{"id": "development", "qualified": True, "initial_pressure_sha256": "hash"}],
        }
        def validate(candidate):
            probe.provenance(candidate, "lab", "patch", ["development"], True, {"development": "hash"})
        validate(report)
        for owner, key, value in [("tool", "patch_commit", "stale"), ("tool", "patch_dirty", "true"),
                                  ("case", "qualified", False), ("case", "initial_pressure_sha256", "changed"),
                                  ("case", "id", "test-case")]:
            candidate = copy.deepcopy(report)
            (candidate["cases"][0] if owner == "case" else candidate[owner])[key] = value
            with self.assertRaises(ValueError):
                validate(candidate)


if __name__ == "__main__":
    unittest.main()
