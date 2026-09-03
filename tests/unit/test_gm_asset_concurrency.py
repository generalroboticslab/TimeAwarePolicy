from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GmAssetConcurrencyTest(unittest.TestCase):
    def test_task_uses_checked_in_assets_without_runtime_rewrites(self):
        source = (
            ROOT / "envs" / "isaacgymenvs" / "tasks" / "franka_gm_pour.py"
        ).read_text()
        setup = source[
            source.index("# Create cups"):source.index("# Create cupA asset")
        ]
        self.assertIn('"urdf/procedural/cup.urdf"', setup)
        self.assertIn('"urdf/procedural/cupB.urdf"', setup)
        self.assertNotIn("create_hollow_cylinder(", setup)
        self.assertNotIn("create_hollow_cylinder_mesh(", setup)

    def test_checked_in_assets_exist_and_are_nonempty(self):
        asset_dir = ROOT / "envs" / "assets" / "urdf" / "procedural"
        for name in ("cup.urdf", "cupB.urdf", "real_cup.stl"):
            path = asset_dir / name
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
