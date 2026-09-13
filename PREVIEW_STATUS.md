# Preview status

This is an internal review candidate. No public anonymous URL has been created for it.

## Checks performed

- 543 selected source files verified against the pinned source's Git blob hashes before copying.
- 15 project Python files had known project-identifying text or machine-specific path strings normalized. An AST comparison confirmed no changes outside those string substitutions and comments. This is a structural check, not an experimental equivalence proof.
- The bundled custom `diffusers` runtime and its packaging/license files were preserved byte-for-byte from the selected source.
- 515 Python files passed parsing/compilation without execution or bytecode output.
- Known source-owner/repository identifiers and common credential patterns were scanned; no matches remained in the candidate. This cannot establish complete anonymity, especially for unknown author names or distinctive content.
- 24 selected modules imported successfully, including this preview's custom `diffusers`, `ags`, DiT/SD entries and evaluation entry. The four core AGS classes were present.
- Four CLI help checks passed: `gen_single_config.py`, `quick_eval_v3.py`, `sample_ags.py`, `sample_ags_sd.py`.
- `uv pip check` reported all 65 installed packages compatible in the existing reference environment. This does not validate installation or dependency resolution from this preview's new requirements file.

Runtime checks used an existing Windows Python 3.10.20 environment with torch 2.7.1+cu128 and torchvision 0.22.1+cu128. Preview source paths were placed first; networking was blocked. The custom `diffusers` import reported 0.30.0.dev0. These checks do not verify fresh editable installation.

## Not verified / required before publication

- Clean installation and fully locked dependencies.
- End-to-end data loading, model loading, GPU generation, evaluation, or reproduction of any paper table.
- Windows spawn compatibility of the retained feature loader; see the README.
- Coverage of every optional script or optional third-party pipeline dependency.
- Manual anonymity review, including names, attributions, external links and distinctive content. Legitimate third-party credit must remain.
- Final inclusion scope and complete redistribution/license notices; see `THIRD_PARTY_NOTICES.md`.
- Hosted anonymous repository behavior and access without login.

The source snapshot, source-to-preview mapping, detailed test logs and private review notes are intentionally outside this candidate. Do not attach those private materials to a blinded submission.
