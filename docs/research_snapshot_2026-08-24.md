# Research workspace snapshot — 2026-08-24

This snapshot preserves the complete local research workspace associated with the
`experiment-v3-rebuild` branch before local cleanup.

## Source state

- Repository: `o0SilentStorm0o/VQE-in-GAN`
- Branch: `experiment-v3-rebuild`
- Workspace source commit: `9dd9ba39d41b7196d30666f87307b32f03de32c7`
- Release tag: `research-snapshot-2026-08-24`
- Release URL:
  `https://github.com/o0SilentStorm0o/VQE-in-GAN/releases/tag/research-snapshot-2026-08-24`

The GitHub branch and release tag preserve the Git-tracked source, tests, protocols, and
reported results. The attached split archive additionally preserves every file in the local
workspace except the `.git` directory, including:

- all raw and diagnostic experiment outputs under `runs/`;
- all model checkpoints and full metric traces;
- the cached MNIST dataset under `data/`;
- the local Python environment under `.venv/`;
- the preprint `2512.12581v2.pdf` and its rendered working pages;
- local test, lint, bytecode, and build caches; and
- the checked-out tracked files at the workspace source commit above.

The archive contains 26,699 entries and expands to 3,147,294,720 bytes. It is split into three
assets so every file stays below GitHub's per-release-asset size limit.

## Release assets

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-aa` | 891,289,600 | `d9fef96466c72db84cce8df695231201216c38cc8583055e46410717fbb8809a` |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-ab` | 891,289,600 | `07c737cefdb9b5692a465974917e868e31b1da866f6bedc4d82f501f926020d2` |
| `vqe-in-gan-workspace-2026-08-24.tar.zst.part-ac` | 506,010,622 | `325858be177c11ee32b2ff6938547879c457de2bae54a865f08e2476fb221c05` |

`research_snapshot_2026-08-24.sha256` is also attached to the release and contains the same
checksums in a format accepted by `shasum -a 256 -c`.

## Restore procedure

Clone the source branch and download all four release assets into the repository root. Then run:

```bash
shasum -a 256 -c research_snapshot_2026-08-24.sha256
cat vqe-in-gan-workspace-2026-08-24.tar.zst.part-* \
  | zstd -dc \
  | tar -xf -
```

The archive is rooted at the repository directory (`./`), so it must be extracted from the clone's
root. The snapshot intentionally excludes `.git`; the clone supplies the Git metadata and complete
committed history.

## Verification performed before upload

- all archive parts passed SHA-256 hashing;
- concatenating the parts passed `zstd -t` integrity verification;
- the archive inventory contains no `.git` entries;
- the inventory includes 594 `runs/` entries, 25,701 `.venv/` entries, 11 `data/` entries, and the
  preprint PDF; and
- a high-confidence secret scan found no access token, API key, or private key marker.

Raw provenance files retain machine-local paths because this is an exact research snapshot. They do
not contain the authentication material used to upload the archive.
