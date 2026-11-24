# Not included in this repository, and why

The repository is built from an explicit allowlist; everything else in the research materials is excluded by
default. This list names the categories that were considered and deliberately left out. It gives no
paths, values or identifiers from the excluded material.

| Category | Reason |
|---|---|
| Fingerprint images (raw, degraded, restored, synthetic-from-real) | biometric data; distribution bounded by participant consent |
| Subject-level split files, per-patch manifests, subject-provenance audits | carry pseudonymous subject indices linkable to images |
| Per-image score files (detector outputs per test image) | each row carries a subject index and finger label |
| Experiment execution logs, run directories, test-access logs | contain local paths and execution history; not needed to use the methods |
| Internal numerical provenance audits (value → source file and row filter) | internal verification records of the authors; kept in full privately |
| Internal claim audit, review records, project documentation and planning notes | internal research records |
| Model weights (all detectors) | release not decided |
| D3 detector code and the other screened third-party detectors | third-party code; D3 and LTD have no licence file; others not reviewed for redistribution |
| Third-party pretrained weights (CLIP, Restormer, upstream screened checkpoints) | obtainable from their own sources; not ours to redistribute |
| The learned restorer of the research code and its checkpoint | depends on a separate model not cleared for release |
| Dataset-building code and data manifests | contain local paths, subject indices and dataset provenance records |
| Training and evaluation drivers of the research code | tied to private data paths and per-image score files; not runnable without them |
| Face-study and public-pairing study code | depend on third-party corpora and their terms; not reviewed for release |
| Paper sources, generated tables and macros | the manuscript is distributed through the journal process, not this repository; the macro audit is internal |
| E060 / E061 access logs, run provenance records and test-read authorisation files | contain local paths, host names and internal authorisation records; the pinned script hashes are restated in `README.md` |
| E060 / E061 validation-only pilot outputs | code exercises on a circular validation pool; not evidence |
| Figure scripts of an earlier, unsubmitted write-up of the detector-level study | figure numbering does not match this manuscript; the underlying aggregates are kept in `data/aggregate/` |
| Records of later, unexecuted study plans | internal planning; no results exist |
| E062 appearance-label table, its source file and any capture-to-class mapping | per-capture labels of identifiable fingerprint captures; the rule is released instead |
| E062 metadata audit, finger-by-class feasibility and label-feature discrepancy audit outputs | contain small finger-by-class cells or private folder structure |
| E062 governance records (PI decisions, provenance statement, scope-change record, authorisation, run lock, access log, preflight and formal runner) | internal authorisation and signature records |
| E068–E070 evaluation drivers (enhancement build, detector/matcher scoring, external-window build, NFIQ 2 scoring), E066 code | tied to private data paths, vendored third-party code and per-image outputs; E066 (an instrument screen that ended without a test read) is not reported in the manuscript |
| Vendored third-party code and weights used by E066/E068 (FpEnhancer, RINE, SPAI, OpenAI CLIP) and the NIST NFIQ 2 binary | not ours to redistribute; pinned by commit/hash in the experiment records and obtainable from their sources |
| FVC2000–2004 and NIST SD302 images, window crops, per-capture NFIQ 2 scores, per-capture window manifests | third-party biometric data; SD302 terms forbid redistribution and require acknowledging the NIST Nail to Nail Fingerprint Challenge; only device-level aggregates are released |
| E063–E071 pilot outputs, access logs, run provenance, gate records with local paths, authorisation files and score dumps | internal records or per-image outputs; the pinned script hashes are restated in `scripts/verify_frozen_sources.py` and `RELEASE_MANIFEST_C1.json` |
| E068 per-image enhancement statistics | per-image records |
| Credentials, tokens, keys, host names | never released (none found in this repository) |

Kept privately and unchanged: every excluded file stays in the authors' research repository, including the
numerical provenance audits and the frozen result files they verify.
