import pytest

from core.evaluation import visualization


def test_rerun_020_uses_legacy_auto_space_views(monkeypatch):
    def legacy_blueprint(*parts, auto_space_views=None, collapse_panels=False):
        return parts, auto_space_views, collapse_panels

    monkeypatch.setattr(visualization.rrb, "Blueprint", legacy_blueprint)
    assert visualization._blueprint_auto_view_option() == {
        "auto_space_views": False
    }


def test_rerun_021_uses_renamed_auto_views(monkeypatch):
    def current_blueprint(*parts, auto_views=None, collapse_panels=False):
        return parts, auto_views, collapse_panels

    monkeypatch.setattr(visualization.rrb, "Blueprint", current_blueprint)
    assert visualization._blueprint_auto_view_option() == {"auto_views": False}


@pytest.mark.parametrize(
    ("timeaware_layout", "simple_layout"),
    ((False, False), (True, False), (True, True)),
)
def test_blueprint_layout_builds_with_installed_rerun(
    timeaware_layout, simple_layout
):
    visualizer = object.__new__(visualization.RerunVis)
    visualizer.timeaware_layout = timeaware_layout
    visualizer.simple_layout = simple_layout
    visualizer.joint_paths = ["Joint_0"]
    blueprint = visualizer._build_blueprint(5.0)
    assert isinstance(blueprint, visualization.rrb.Blueprint)
