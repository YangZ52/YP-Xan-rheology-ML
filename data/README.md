# Data Directory

Raw and processed data are not committed to this repository by default.

Recommended structure for local work:

```text
data/
  raw/          Original rheology workbooks
  interim/      Parsed intermediate files
  processed/    Public/anonymized ML-ready tables for release
```

For public release or paper review, provide either:

1. a DOI/link to a data repository containing the raw or ML-ready files, or
2. anonymized ML-ready CSV/XLSX files under `data/processed/` with scripts updated to read from this directory.

