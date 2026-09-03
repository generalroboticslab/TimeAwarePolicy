"""Checks for the small, self-contained public artifact bundle."""

from tests.release.verify_artifacts import DEFAULT_MANIFEST, verify


def test_public_artifact_manifest_verifies() -> None:
    result = verify(DEFAULT_MANIFEST)
    assert result["status"] == "passed"
    assert result["tasks"] == 3
    assert result["files_checked"] == 24
