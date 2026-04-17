"""Configuration loading and resolution utilities."""

import os

import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML task config and resolve relative paths.

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed config dict with ``root_path`` resolved to an absolute path.
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(config_path)
    if "root_path" in cfg:
        cfg["root_path"] = os.path.abspath(
            os.path.join(yaml_dir, cfg["root_path"])
        )
    return cfg


def apply_scatter_config(cfg: dict) -> None:
    """Override plane config with scatter_area from grasp config (in-place).

    If ``cfg['grasp']['scatter_area']`` is defined AND
    ``cfg['grasp']['use_scatter_area']`` is not False, the plane position
    and scale are updated so that replicator scatter constrains objects to
    the specified area.
    """
    grasp_cfg = cfg.get("grasp", {})
    use_scatter = grasp_cfg.get("use_scatter_area", True)  # default True
    scatter_cfg = grasp_cfg.get("scatter_area", None)

    if scatter_cfg is not None and use_scatter:
        sc = scatter_cfg["center"]
        ss = scatter_cfg["size"]
        cfg["plane"]["plane_position"] = [[sc[0], sc[1], sc[2]]]
        cfg["plane"]["plane_scale"] = [[ss[0], ss[1], 0.1]]
        print(f"[Scatter] Scatter area: center={sc}, size={ss}")
        print(
            f"[Scatter] x=[{sc[0]-ss[0]/2:.2f}, {sc[0]+ss[0]/2:.2f}], "
            f"y=[{sc[1]-ss[1]/2:.2f}, {sc[1]+ss[1]/2:.2f}], z={sc[2]:.2f}"
        )
    elif scatter_cfg is not None and not use_scatter:
        print("[Scatter] use_scatter_area=false, skipping scatter_area override, using plane default config")
        if "plane" in cfg:
            pp = cfg["plane"]["plane_position"][0]
            ps = cfg["plane"]["plane_scale"][0]
            print(f"[Scatter] Using default plane config: center={pp}, scale={ps}")
    else:
        if "plane" in cfg:
            pp = cfg["plane"]["plane_position"][0]
            ps = cfg["plane"]["plane_scale"][0]
            print(f"[Scatter] Using default plane config: center={pp}, scale={ps}")
        else:
            print("[Scatter] Plane config not defined, please check config")
