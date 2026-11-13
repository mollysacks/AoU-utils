"""CNV calls.

- `recurrent`: person x locus copy-number matrix for named recurrent CNVs (e.g. 16p11.2), and
  grouping people by the CNVs they carry.
- `segments`: segment-level calls (one row per CNV: person, chrom, start, end, DEL/DUP), filtering,
  and per-person burden (Mb and number of CNVs).

Import the modules directly: `from aou_utils.cnv import recurrent, segments`.
"""
