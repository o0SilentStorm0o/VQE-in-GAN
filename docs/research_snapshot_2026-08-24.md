# Research workspace snapshot — 2026-08-24

This snapshot preserves the complete local research workspace associated with the
`experiment-v3-rebuild` branch before local cleanup.

## Source state

- Repository: `o0SilentStorm0o/VQE-in-GAN`
- Branch: `experiment-v3-rebuild`
- Workspace content commit: `805840ab8fe00ad46ef4e1338c79745cf93f0db0`
- Contributor identity: `o0SilentStorm0o <davidstrnadel@seznam.cz>`
- Release tag: `research-snapshot-2026-08-24`
- Release URL:
  `https://github.com/o0SilentStorm0o/VQE-in-GAN/releases/tag/research-snapshot-2026-08-24`

The GitHub branch and release tag preserve the Git-tracked source, tests, protocols, result
summaries, current V3 README, and
[revision identity map](revision_identity_map_2026-08-24.md). The attached split archive
additionally preserves the local files that ordinary Git intentionally excludes:

- all raw and diagnostic experiment outputs under `runs/`;
- all model checkpoints and full metric traces;
- the cached MNIST dataset under `data/`;
- the local Python environment under `.venv/`;
- the preprint `2512.12581v2.pdf` and its rendered working pages under `tmp/`;
- local test, lint, bytecode, and build caches; and
- the checked-out tracked files at the workspace content commit above.

The archive excludes only:

1. `.git/`, because a fresh clone supplies the complete committed history; and
2. `docs/research_snapshot_2026-08-24.md` plus
   `docs/research_snapshot_2026-08-24.sha256`, because their checksums describe the archive
   itself and both files are preserved by the release-tagged Git commit.

This avoids a self-referential checksum while leaving the restored clone complete. The archive
contains 26,700 entries, expands to 3,147,304,960 bytes, and is split into three assets below
GitHub's per-release-asset size limit. Their combined compressed size is 2,288,579,828 bytes.

## Release assets

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-aa` | 891,289,600 | `ab071fbeb3b54e523569b8ad5eb14fdcae4fb5ab49de004b14ee58dfcb70d841` |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-ab` | 891,289,600 | `802c746b6a9f81c855dda13eee48797097a916c23d4a63f4a126f63a9de5c043` |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-ac` | 506,000,628 | `0ad04ead8b8fd3efcd602d3feebda9ff7eb57a081578778df13faefdfacf7f0f` |

`research_snapshot_2026-08-24.sha256` is attached separately to the release and contains the
same values in a format accepted by `shasum -a 256 -c`.

## Restore procedure

Clone the V3 source branch and download all release assets into the repository root:

```bash
git clone --branch experiment-v3-rebuild \
  https://github.com/o0SilentStorm0o/VQE-in-GAN.git
cd VQE-in-GAN
gh release download research-snapshot-2026-08-24
shasum -a 256 -c research_snapshot_2026-08-24.sha256
cat vqe-in-gan-workspace-2026-08-24.tar.zst.part-* \
  | zstd -dc \
  | tar -xf -
```

The archive is rooted at the repository directory (`./`). Extract it only from the fresh clone's
root. It does not contain `.git` and therefore cannot overwrite repository history or remote
configuration.

The preserved `.venv` is useful for an exact same-machine snapshot but is not guaranteed to be
portable across operating systems or Python installations. On another system, recreate it from
the authoritative lock file with `uv sync --extra dev`.

## Verification performed before upload

- all three archive parts passed SHA-256 hashing;
- concatenating the parts passed `zstd -t` integrity verification;
- the decompressed stream was measured at exactly 3,147,304,960 bytes;
- the archive inventory contains no `.git` or self-referential manifest entries;
- the inventory contains 594 `runs/` entries, 25,701 `.venv/` entries, 11 `data/` entries,
  the V3 README, the revision map, the preprint PDF, and six `tmp/` entries; and
- a high-confidence credential scan found no private key or real access token.

The token-pattern scan produced one false positive inside an embedded Base64 font shipped by
Pillow 12.3.0. Inspection confirmed that it is package data rather than authentication material.
Raw provenance files retain machine-local paths because this is an exact research snapshot; they
do not contain the authentication material used to upload it.

Local deletion is safe only after the release upload, remote checksums, and an independent
download-and-restore audit have all completed successfully.
