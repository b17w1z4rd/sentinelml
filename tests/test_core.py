import json
import tempfile
import unittest
from pathlib import Path
from sentinelml.core import FEATURES, generate, load_csv, score, split_groups, train


class ExperimentTests(unittest.TestCase):
    def test_episode_splits_and_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            generate(root / "data.csv", episodes=300)
            x, y, groups = load_csv(root / "data.csv")
            train_idx, val_idx, test_idx = split_groups(x, y, groups)
            subsets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
            self.assertFalse(subsets[0] & subsets[1] | subsets[0] & subsets[2] | subsets[1] & subsets[2])
            report = train(root / "data.csv", root / "model.joblib", root / "report.json")
            self.assertGreater(report["test"]["average_precision"], .3)
            result = score(root / "model.joblib", dict(zip(FEATURES, x[0])))
            self.assertIn("alert", result)
            self.assertEqual(json.loads((root / "report.json").read_text())["threshold"], result["threshold"])

    def test_reject_invalid_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(Path(d) / "data.csv", episodes=80)
            text = p.read_text().replace("episode-00000,", "episode-00000,2,", 1)
            p.write_text(text)
            with self.assertRaises(ValueError):
                load_csv(p)


if __name__ == "__main__":
    unittest.main()
