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
| Credentials, tokens, keys, host names | never released (none found in this repository) |

Kept privately and unchanged: every excluded file stays in the authors' research repository, including the
numerical provenance audits and the frozen result files they verify.
