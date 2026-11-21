from pathlib import Path

from scripts.validate_blocks_maps import active_map


def test_active_map_preserves_nested_content_path(tmp_path: Path):
    project = tmp_path / "Blocks.uproject"
    log = tmp_path / "Saved" / "Logs" / "Blocks.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        'Cmd: MAP LOAD FILE="D:/AirSim/Blocks/Content/FlyingCPP/Maps/FlyingExampleMap.umap" '
        "TEMPLATE=0 SHOWPROGRESS=1\n",
        encoding="utf-8",
    )

    assert active_map(project) == "/Game/FlyingCPP/Maps/FlyingExampleMap"


def test_active_map_last_mixed_log_record_wins(tmp_path: Path):
    project = tmp_path / "Blocks.uproject"
    log = tmp_path / "Saved" / "Logs" / "Blocks.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "LoadMap(/Game/old/OldMap)\n"
        'MAP LOAD FILE="D:/AirSim/Blocks/Content/FlyingCPP/Maps/FlyingExampleMap.umap"\n'
        "Created PIE world by copying editor world from /Game/maze.maze\n",
        encoding="utf-8",
    )

    assert active_map(project) == "/Game/maze"
