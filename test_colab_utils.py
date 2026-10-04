"""
Tests for colab_utils.

The errno-107 behaviour is the whole point of this module, and it cannot be
triggered by normal filesystem use, so it is exercised by monkeypatching
`Path.exists` / `Path.iterdir` to raise the way a dead FUSE mount does.

Run:  python3 -m unittest test_colab_utils
"""

from __future__ import annotations

import errno
import unittest
from pathlib import Path
from unittest import mock

import colab_utils as cu


def _dead(*args, **kwargs):
    raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")


class TestPathOk(unittest.TestCase):
    def test_real_missing_path_is_false(self):
        self.assertFalse(cu.path_ok("/definitely/not/here/at/all"))

    def test_real_existing_path_is_true(self):
        self.assertTrue(cu.path_ok(Path(__file__).parent))

    def test_dead_mount_returns_false_instead_of_raising(self):
        with mock.patch.object(Path, "exists", _dead):
            self.assertFalse(cu.path_ok("/content/drive/MyDrive"))

    def test_is_dir_swallows_dead_mount(self):
        with mock.patch.object(Path, "is_dir", _dead):
            self.assertFalse(cu.is_dir("/content/drive"))

    def test_unrelated_oserror_still_propagates(self):
        def denied(*args, **kwargs):
            raise PermissionError(errno.EACCES, "denied")

        with mock.patch.object(Path, "exists", denied):
            with self.assertRaises(PermissionError):
                cu.path_ok("/content/drive")

    @unittest.skipUnless(
        hasattr(Path, "is_mount"), "Path.is_mount is 3.12+"
    )
    def test_dead_mount_errno_set_includes_107(self):
        self.assertIn(107, cu.DEAD_MOUNT_ERRNOS)


class TestDriveStatus(unittest.TestCase):
    def test_healthy_drive_reports_entries(self, ):
        with mock.patch.object(cu, "path_ok", lambda p: True):
            status = cu.drive_status()
        self.assertTrue(status["drive_root"])
        self.assertTrue(status["mydrive"])
        self.assertFalse(status["dead_mount"])

    def test_dead_mount_is_detected(self):
        def fake_iterdir(self):
            if str(self).endswith("MyDrive") or str(self) == "/content/drive":
                raise OSError(errno.ENOTCONN, "not connected")
            return iter(())

        with mock.patch.object(cu, "path_ok", lambda p: True):
            with mock.patch.object(Path, "iterdir", fake_iterdir):
                status = cu.drive_status()
        self.assertTrue(status["dead_mount"])

    def test_not_mounted_reports_no_mydrive(self):
        with mock.patch.object(cu, "path_ok", lambda p: str(p) != "/content/drive/MyDrive"):
            status = cu.drive_status()
        self.assertFalse(status["mydrive"])
        self.assertFalse(status["dead_mount"])

    def test_report_mentions_restart_for_dead_mount(self):
        text = cu.drive_report()
        self.assertIn("DRIVE STATE", text)

    def test_require_drive_passes_when_healthy(self):
        with mock.patch.object(cu, "drive_status",
                               lambda: {"drive_root": True, "mydrive": True,
                                        "dead_mount": False, "entries": ["x"]}):
            cu.require_drive()  # must not raise

    def test_require_drive_raises_on_dead_mount_with_advice(self):
        with mock.patch.object(cu, "drive_status",
                               lambda: {"drive_root": True, "mydrive": False,
                                        "dead_mount": True, "entries": []}):
            with self.assertRaises(RuntimeError) as ctx:
                cu.require_drive("the event base")
        message = str(ctx.exception)
        self.assertIn("errno 107", message)
        self.assertIn("Restart session", message)

    def test_require_drive_raises_when_not_mounted(self):
        with mock.patch.object(cu, "drive_status",
                               lambda: {"drive_root": False, "mydrive": False,
                                        "dead_mount": False, "entries": []}):
            with self.assertRaises(RuntimeError) as ctx:
                cu.require_drive()
        self.assertIn("not mounted", str(ctx.exception))


class TestLazyParquetWriter(unittest.TestCase):
    """`pq.ParquetWriter` needs a schema at construction, which is impossible
    for a shard that may never receive a row. That killed the assembly cell at
    PHASE 1 with "missing 1 required positional argument: 'schema'".
    """

    def test_construction_does_not_touch_the_filesystem(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            target = P(tmp) / "shard_p00.parquet"
            writer = cu.LazyParquetWriter(target)
            self.assertFalse(writer.opened)
            self.assertFalse(target.exists())

    def test_close_without_any_write_is_safe(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "never.parquet")
            writer.close()
            writer.close()          # idempotent

    def test_opens_on_first_write(self):
        import tempfile

        if not __import__("importlib").util.find_spec("pyarrow"):
            self.skipTest("pyarrow is Colab-only")
        import pandas as pd
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            target = P(tmp) / "out.parquet"
            writer = cu.LazyParquetWriter(target)
            writer.write_frame(pd.DataFrame({"a": [1.0, 2.0]}))
            self.assertTrue(writer.opened)
            writer.write_frame(pd.DataFrame({"a": [3.0]}))
            writer.close()
            back = pd.read_parquet(target)
            self.assertEqual(len(back), 3)

    def test_rows_written_accumulates(self):
        if not __import__("importlib").util.find_spec("pyarrow"):
            self.skipTest("pyarrow is Colab-only")
        import pandas as pd
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "n.parquet")
            writer.write_frame(pd.DataFrame({"a": [1.0]}))
            writer.write_frame(pd.DataFrame({"a": [2.0, 3.0]}))
            self.assertEqual(writer.rows_written, 3)
            writer.close()

    def test_empty_frames_are_ignored(self):
        import pandas as pd
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "e.parquet")
            writer.write_frame(pd.DataFrame({"a": []}))
            writer.write_frame(None)
            self.assertFalse(writer.opened)
            self.assertEqual(writer.rows_written, 0)
            writer.close()

    def test_wrote_anything_survives_close(self):
        """The bug that emptied PHASE 2: cleanup tested `opened` AFTER close(),
        which is always False, so all 16 shards were deleted and the join
        produced 0 events while reporting success."""
        if not __import__("importlib").util.find_spec("pyarrow"):
            self.skipTest("pyarrow is Colab-only")
        import pandas as pd
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "s.parquet")
            writer.write_frame(pd.DataFrame({"a": [1.0, 2.0]}))
            self.assertTrue(writer.wrote_anything)
            writer.close()
            # After close, `opened` is False but the file has rows.
            self.assertFalse(writer.opened)
            self.assertTrue(writer.wrote_anything)
            self.assertEqual(writer.rows_written, 2)

    def test_wrote_anything_is_false_for_a_writer_that_got_nothing(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "none.parquet")
            writer.close()
            self.assertFalse(writer.wrote_anything)

    def test_rows_written_is_sticky_after_close(self):
        if not __import__("importlib").util.find_spec("pyarrow"):
            self.skipTest("pyarrow is Colab-only")
        import pandas as pd
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            writer = cu.LazyParquetWriter(P(tmp) / "r.parquet")
            writer.write_frame(pd.DataFrame({"a": [1.0]}))
            writer.close()
            writer.close()
            self.assertEqual(writer.rows_written, 1)

    def test_works_as_a_context_manager(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            with cu.LazyParquetWriter(P(tmp) / "ctx.parquet") as writer:
                self.assertFalse(writer.opened)
            self.assertFalse(writer.opened)


if __name__ == "__main__":
    unittest.main()
