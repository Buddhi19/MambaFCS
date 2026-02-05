import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _maybe_load_dotenv(repo_root: str) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        dotenv_path = os.path.join(repo_root, ".env")
        if not os.path.isfile(dotenv_path):
            return

        parsed: Dict[str, str] = {}
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    parsed[key] = value

        for key, value in parsed.items():
            if key not in os.environ:
                os.environ[key] = value
        return

    dotenv_path = os.path.join(repo_root, ".env")
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        load_dotenv()


def _expand_str(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _expand_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_tree(v) for v in obj]
    if isinstance(obj, str):
        return _expand_str(obj)
    return obj


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    path = _expand_str(path)
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _resolve_paths(paths: Iterable[str], base_dir: str) -> List[str]:
    resolved: List[str] = []
    for p in paths:
        rp = _resolve_path(p, base_dir)
        if rp is None:
            continue
        resolved.append(rp)
    return resolved


def _read_list_file(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                items.append(item)
    return items


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to run this script. Install it with `pip install pyyaml`."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping (dict). Got: {type(data)!r}")
    return data


def _required(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def _maybe_warn_unexpanded(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    unresolved: List[str] = []
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str) and ("${" in v or "$" in v):
            unresolved.append(k)
    if unresolved:
        joined = ", ".join(unresolved)
        raise RuntimeError(
            f"Unexpanded environment variable(s) in: {joined}. "
            "Set the env vars (or update the YAML to absolute paths)."
        )


def _build_args(cfg: Dict[str, Any], repo_root: str) -> SimpleNamespace:
    dataset_root = _resolve_path(cfg.get("dataset_root"), repo_root)
    dataset_base = dataset_root or repo_root

    train_dataset_path = _resolve_path(str(_required(cfg, "train_dataset_path")), dataset_base)
    test_dataset_path = _resolve_path(str(_required(cfg, "test_dataset_path")), dataset_base)

    train_list_paths_raw = cfg.get("train_data_list_paths") or []
    if isinstance(train_list_paths_raw, str):
        train_list_paths_raw = [train_list_paths_raw]
    if not isinstance(train_list_paths_raw, list):
        raise ValueError("`train_data_list_paths` must be a string or list of strings.")

    test_list_paths_raw = cfg.get("test_data_list_path")
    if test_list_paths_raw is None:
        test_list_paths_raw = cfg.get("test_data_list_paths")  # allow alias
    if test_list_paths_raw is None:
        test_list_paths_raw = []
    if isinstance(test_list_paths_raw, str):
        test_list_paths_raw = [test_list_paths_raw]
    if not isinstance(test_list_paths_raw, list):
        raise ValueError("`test_data_list_path(s)` must be a string or list of strings.")

    train_list_paths = _resolve_paths([str(p) for p in train_list_paths_raw], dataset_base)
    test_list_paths = _resolve_paths([str(p) for p in test_list_paths_raw], dataset_base)

    train_data_name_list = cfg.get("train_data_name_list")
    if train_data_name_list is None:
        if not train_list_paths:
            raise KeyError("Provide `train_data_name_list` or `train_data_list_paths`.")
        train_data_name_list = []
        for p in train_list_paths:
            train_data_name_list.extend(_read_list_file(p))
    if not isinstance(train_data_name_list, list):
        raise ValueError("`train_data_name_list` must be a list of strings.")

    test_data_name_list = cfg.get("test_data_name_list")
    if test_data_name_list is None:
        if not test_list_paths:
            raise KeyError("Provide `test_data_name_list` or `test_data_list_path`.")
        test_data_name_list = []
        for p in test_list_paths:
            test_data_name_list.extend(_read_list_file(p))
    if not isinstance(test_data_name_list, list):
        raise ValueError("`test_data_name_list` must be a list of strings.")

    cfg_path = _resolve_path(str(_required(cfg, "cfg")), repo_root)
    pretrained_weight_path = _resolve_path(str(_required(cfg, "pretrained_weight_path")), repo_root)
    model_param_path = _resolve_path(str(_required(cfg, "model_param_path")), repo_root)

    opts = cfg.get("opts")
    if opts is not None and not isinstance(opts, list):
        raise ValueError("`opts` must be null or a list of strings (yacs-style merge list).")

    max_iters = int(cfg.get("max_iters", 0))
    batch_size = int(cfg.get("batch_size", 1))
    if max_iters <= 0:
        raise ValueError("`max_iters` must be > 0.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")

    args = SimpleNamespace(
        cfg=cfg_path,
        opts=opts,
        pretrained_weight_path=pretrained_weight_path,
        dataset=str(_required(cfg, "dataset")),
        type=str(cfg.get("type") or "train"),
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        shuffle=bool(cfg.get("shuffle", True)),
        batch_size=batch_size,
        crop_size=int(cfg.get("crop_size", 256)),
        train_data_name_list=train_data_name_list,
        test_data_name_list=test_data_name_list,
        start_iter=int(cfg.get("start_iter", 0)),
        cuda=bool(cfg.get("cuda", True)),
        max_iters=max_iters,
        model_type=str(cfg.get("model_type") or "MambaSCD_base"),
        model_param_path=model_param_path,
        resume=_resolve_path(cfg.get("resume"), repo_root),
        optim_path=_resolve_path(cfg.get("optim_path"), repo_root),
        scheduler_path=_resolve_path(cfg.get("scheduler_path"), repo_root),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        num_classes=int(cfg.get("num_classes", 2)),
        model_saving_name=str(_required(cfg, "model_saving_name")),
    )

    if args.resume is not None and (args.optim_path is None or args.scheduler_path is None):
        raise ValueError(
            "When `resume` is set, both `optim_path` and `scheduler_path` must also be set "
            "(they are required by `train_MambaSCD.Trainer`)."
        )

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="MambaFCS training entrypoint (YAML-driven).")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a YAML config (e.g. MambaFCS/configs/train_LANDSAT.yaml).",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Optional yacs-style overrides appended to YAML `opts`.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved args and exit.")
    cli = parser.parse_args()

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _maybe_load_dotenv(repo_root)

    cfg_path = os.path.abspath(cli.config)
    cfg = _expand_tree(_load_yaml(cfg_path))

    _maybe_warn_unexpanded(
        cfg,
        keys=(
            "dataset_root",
            "train_dataset_path",
            "test_dataset_path",
            "cfg",
            "pretrained_weight_path",
            "model_param_path",
            "resume",
            "optim_path",
            "scheduler_path",
        ),
    )

    if cli.opts:
        yaml_opts = cfg.get("opts")
        merged: List[str] = []
        if isinstance(yaml_opts, list):
            merged.extend([str(x) for x in yaml_opts])
        merged.extend([str(x) for x in cli.opts])
        cfg["opts"] = merged

    cuda_device = cfg.get("cuda_device")
    if cuda_device is not None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(int(cuda_device))
        except Exception:
            pass

    args = _build_args(cfg, repo_root)

    if cli.dry_run:
        print("Resolved training args:")
        for k, v in sorted(vars(args).items()):
            if k in {"train_data_name_list", "test_data_name_list"}:
                print(f"  {k}: <{len(v)} items>")
            else:
                print(f"  {k}: {v}")
        return

    import torch
    from MambaFCS.changedetection.script import train_MambaSCD

    torch.cuda.empty_cache()
    trainer = train_MambaSCD.Trainer(args)
    trainer.training()

    if bool(cfg.get("do_validation", False)):
        trainer.validation()


if __name__ == "__main__":
    main()
