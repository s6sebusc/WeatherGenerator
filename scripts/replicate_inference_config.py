#!/usr/bin/env python3
"""Mirror the config creation path used by WeatherGenerator inference.

This mirrors the runtime sequence in `src/weathergen/run_train.py`:

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

The core point is that the model config is not manually assembled; it is loaded from
an earlier run's JSON, then merged with the private config and CLI/config overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from weathergen.common import config


def build_inference_config(
    from_run_id: str,
    mini_epoch: int = -1,
    run_id: str | None = None,
    private_config: str | Path | None = None,
    base_config: str | Path | None = None,
    extra_config_paths: list[str | Path] | None = None,
    cli_options: list[str] | None = None,
    reuse_run_id: bool = False,
):
    """Build the exact config object that the CLI inference path produces."""
    cli_overwrite = config.from_cli_arglist(cli_options or [])
    extra_config_paths = extra_config_paths or []

    cf = config.load_merge_configs(
        Path(private_config) if private_config else None,
        from_run_id,
        mini_epoch,
        Path(base_config) if base_config else None,
        *[Path(p) for p in extra_config_paths],
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, run_id, reuse_run_id)
    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run-id", required=True, help="Run id to load and continue from")
    parser.add_argument("--mini-epoch", type=int, default=-1, help="Checkpoint epoch to use, default -1")
    parser.add_argument("--run-id", help="New run id; if omitted a random one is generated")
    parser.add_argument("--private-config", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--config", action="append", default=[], help="Extra config file(s)")
    parser.add_argument("--option", action="append", default=[], help="OmegaConf CLI override like a.b.c=value")
    parser.add_argument("--reuse-run-id", action="store_true")
    args = parser.parse_args()

    cf = build_inference_config(
        from_run_id=args.from_run_id,
        mini_epoch=args.mini_epoch,
        run_id=args.run_id,
        private_config=args.private_config,
        base_config=args.base_config,
        extra_config_paths=args.config,
        cli_options=args.option,
        reuse_run_id=args.reuse_run_id,
    )

    print(f"Loaded base config from from_run_id={args.from_run_id}")
    print(f"Resulting run_id: {cf.general.run_id}")
    print(f"from_run_id in config: {cf.get('from_run_id')}")
    print(config.format_cf(cf)[:2000])
