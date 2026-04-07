#!/usr/bin/env python3
"""Upload or download ``meta_libero/results`` via Hugging Face Hub.

Requires: ``pip install huggingface_hub`` (not part of the core LIBERO stack).

Authentication (pick one):
  - ``huggingface-cli login`` (stores token in ``~/.cache/huggingface/token``)
  - Environment: ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``

Create an empty dataset repo on https://huggingface.co/new-dataset first, or pass
``--create-repo`` to create it from this script (needs a write token).

Examples (from the ``meta_vlas`` repo root):

  # Upload everything under meta_libero/results to username/my-meta-libero-results
  python meta_libero/scripts/meta_libero_results_hf.py upload \\
    --repo-id username/my-meta-libero-results --create-repo --private

  # Same, then delete local copy (irreversible — use only after verifying the Hub upload)
  python meta_libero/scripts/meta_libero_results_hf.py upload \\
    --repo-id username/my-meta-libero-results --remove-local-after-upload --yes

  # Download snapshot back to ./meta_libero/results (or META_LIBERO_RESULTS_DIR)
  python meta_libero/scripts/meta_libero_results_hf.py download \\
    --repo-id username/my-meta-libero-results

By default, **upload** uses ``HfApi.upload_large_folder()`` (resumable, batched commits, metadata
under ``<local-dir>/.cache/.huggingface/``). Use ``--upload-method folder`` only for small trees or
when you need ``--path-in-repo`` (single ``upload_folder`` commit; may warn/fail on very large dirs).
See https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _default_results_dir() -> Path:
    env = os.getenv("META_LIBERO_RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # meta_libero/scripts -> meta_libero/results
    return (Path(__file__).resolve().parents[1] / "results").resolve()


def _import_hf():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        print(
            "Missing dependency: huggingface_hub\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return HfApi, snapshot_download


def cmd_upload(args: argparse.Namespace) -> None:
    HfApi, _ = _import_hf()
    local = Path(args.local_dir).expanduser().resolve()
    if not local.is_dir():
        raise SystemExit(f"Local directory does not exist or is not a directory: {local}")

    api = HfApi(token=args.token)
    repo_id = args.repo_id
    if args.create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    path_in_repo = (args.path_in_repo or "").strip()
    if args.upload_method == "folder" and args.num_workers is not None:
        raise SystemExit("--num-workers is only valid with --upload-method large")
    if args.upload_method == "large":
        if path_in_repo:
            raise SystemExit(
                "upload_large_folder does not support --path-in-repo (Hub limitation). "
                "Omit --path-in-repo to upload at repo root, or nest files locally under a subfolder, "
                "or use --upload-method folder (single commit; not recommended for huge trees)."
            )
        upload_large = getattr(api, "upload_large_folder", None)
        if upload_large is None:
            raise SystemExit(
                "This huggingface_hub is too old (no upload_large_folder). "
                "Upgrade huggingface_hub or pass --upload-method folder."
            )
        print(
            f"Uploading (large-folder mode) {local} -> {repo_id} ({args.repo_type}) ...\n"
            "Resume: re-run the same command; progress is cached under "
            f"{local / '.cache' / '.huggingface'} (do not delete while uploading)."
        )
        kw: dict = {
            "repo_id": repo_id,
            "folder_path": str(local),
            "repo_type": args.repo_type,
        }
        if args.create_repo:
            kw["private"] = args.private
        if args.num_workers is not None:
            kw["num_workers"] = int(args.num_workers)
        upload_large(**kw)
    else:
        print(
            f"Uploading (single-commit folder mode) {local} -> {repo_id} "
            f"({args.repo_type}) path_in_repo={path_in_repo!r} ..."
        )
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
        )
    print("Upload finished.")

    if args.remove_local_after_upload:
        if not args.yes:
            raise SystemExit(
                "Refusing to remove local directory without --yes "
                "(this permanently deletes your local copy)."
            )
        print(f"Removing local directory: {local}")
        shutil.rmtree(local)
        print("Local directory removed.")


def cmd_download(args: argparse.Namespace) -> None:
    _, snapshot_download = _import_hf()
    dest = Path(args.local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} ({args.repo_type}) -> {dest} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(dest),
        token=args.token,
        local_dir_use_symlinks=False,
    )
    print("Download finished.")


def _add_hub_args(ap: argparse.ArgumentParser) -> None:
    """Repo id / type / token on each subparser so ``upload --repo-id X`` works."""
    ap.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo id, e.g. username/my-meta-libero-results",
    )
    ap.add_argument(
        "--repo-type",
        default="dataset",
        choices=("dataset", "model", "space"),
        help="Hub repo type (default: dataset — good for arbitrary run artifacts)",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF access token (default: env HF_TOKEN or cached login)",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Upload/download meta_libero experiment results via Hugging Face Hub."
    )
    sub = p.add_subparsers(dest="command", required=True)

    u = sub.add_parser("upload", help="Upload a local directory to the Hub")
    _add_hub_args(u)
    u.add_argument(
        "--local-dir",
        type=Path,
        default=_default_results_dir(),
        help=f"Directory to upload (default: {_default_results_dir()})",
    )
    u.add_argument(
        "--upload-method",
        choices=("large", "folder"),
        default="large",
        help=(
            "large = HfApi.upload_large_folder (default; resumable, multi-commit for big trees). "
            "folder = HfApi.upload_folder (single commit; use with --path-in-repo if needed)."
        ),
    )
    u.add_argument(
        "--path-in-repo",
        default="",
        help="Destination path inside the repo (empty = root). Only for --upload-method folder.",
    )
    u.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="Only for --upload-method large: number of parallel workers (Hub default: half of CPU cores).",
    )
    u.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo if it does not exist (needs write token)",
    )
    u.add_argument(
        "--private",
        action="store_true",
        help="When used with --create-repo, create a private repo",
    )
    u.add_argument(
        "--commit-message",
        default="Sync meta_libero results",
        help="Commit message for the upload",
    )
    u.add_argument(
        "--remove-local-after-upload",
        action="store_true",
        help="Delete --local-dir after a successful upload (requires --yes)",
    )
    u.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive actions (e.g. --remove-local-after-upload)",
    )
    u.set_defaults(func=cmd_upload)

    d = sub.add_parser("download", help="Download repo snapshot into a local directory")
    _add_hub_args(d)
    d.add_argument(
        "--local-dir",
        type=Path,
        default=_default_results_dir(),
        help=f"Directory to write (default: {_default_results_dir()})",
    )
    d.set_defaults(func=cmd_download)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
