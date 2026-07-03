"""Tests for the annotator Flask server's HDF5 sample endpoint.

These use Flask's test client (no real socket / browser). Focus: videos are read
from the correct ``observations/video_paths`` group — a regression guard for the
bug where the server read a non-existent ``image_observations`` group and only
rendered videos by accident via the ``<stem>_<cam>.mp4`` filename fallback.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from oopsie_tools.annotation_tool.annotator_server import app, configure_runtime
from oopsie_tools.test.fixtures.make_valid import (
    _write_base_h5,
    _write_video,
    write_valid_episode,
)


@pytest.fixture
def client(tmp_path: Path):
    configure_runtime(
        samples_dir=tmp_path, annotator_name="test_annotator", browse_only=True
    )
    app.config.update(TESTING=True)
    return app.test_client()


def _get_sample(client, rel_path: str) -> dict:
    resp = client.get(f"/api/h5/sample?path={rel_path}")
    assert resp.status_code == 200, resp.data
    return resp.get_json()


def test_video_urls_from_video_paths_group_nested_layout(
    client, tmp_path: Path
) -> None:
    """Video sits in a subdir with no ``<stem>_<cam>.mp4`` sibling, so only a
    correct ``observations/video_paths`` read (not the glob fallback) finds it."""
    nested = tmp_path / "nested"
    (nested / "cam_videos").mkdir(parents=True)
    _write_video(nested / "cam_videos" / "front.mp4", color=(10, 20, 30))
    with h5py.File(nested / "ep.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="ep",
            language_instruction="do the thing",
            camera_video_paths={"front": "cam_videos/front.mp4"},
        )

    data = _get_sample(client, "nested/ep.h5")

    assert list(data["video_urls"].keys()) == ["front"]
    assert data["video_urls"]["front"].endswith("cam_videos/front.mp4")


def test_video_urls_resolved_for_flat_layout(client, tmp_path: Path) -> None:
    """A standard flat episode still resolves its camera video."""
    write_valid_episode(tmp_path, stem="flat")

    data = _get_sample(client, "flat.h5")

    assert "front" in data["video_urls"]
