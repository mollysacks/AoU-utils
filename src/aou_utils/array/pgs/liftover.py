"""Point liftover with a UCSC chain file (e.g. hg19ToHg38.over.chain.gz), vectorized with numpy.

UCSC liftOver chains are netted, so on the source assembly no base is covered by two blocks.
That lets us find each position's block with a binary search.
"""

import gzip

import numpy as np
import pandas as pd

HG19_TO_HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"


def read_chain(path) -> pd.DataFrame:
    """Aligned blocks of a chain file, one row per block, in 0-based half-open coordinates.

    Columns: chrom, start, end (source), q_chrom, q_start, q_size, q_strand. For '-' strand chains
    q_start is on the reverse strand, as in the file.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if not f:
                continue
            if f[0] == "chain":
                chrom, t, q_chrom, q_size, q_strand, q = f[2], int(f[5]), f[7], int(f[8]), f[9], int(f[10])
                continue
            size = int(f[0])
            rows.append((chrom, t, t + size, q_chrom, q, q_size, q_strand))
            if len(f) == 3:
                t += size + int(f[1])
                q += size + int(f[2])
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "q_chrom", "q_start", "q_size", "q_strand"])


def _chrom_name(c) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def lift(chrom, pos, chain: pd.DataFrame) -> pd.DataFrame:
    """Lift 1-based positions. Returns chrom, pos and strand ('+'/'-') per input, NaN where unmapped.

    `chrom` may be '1' or 'chr1'; output chromosomes are named as in the chain file ('chr1').
    """
    chrom = pd.Series(chrom).map(_chrom_name).to_numpy()
    pos0 = np.asarray(pos, dtype=np.int64) - 1
    out_chrom = np.full(len(pos0), None, dtype=object)
    out_pos = np.full(len(pos0), np.nan)
    out_strand = np.full(len(pos0), None, dtype=object)
    for c, blocks in chain.groupby("chrom", sort=False):
        idx = np.flatnonzero(chrom == c)
        if not len(idx):
            continue
        blocks = blocks.sort_values("start")
        starts, ends = blocks["start"].to_numpy(), blocks["end"].to_numpy()
        k = np.searchsorted(starts, pos0[idx], side="right") - 1
        ok = (k >= 0) & (pos0[idx] < ends[np.clip(k, 0, None)])
        idx, k = idx[ok], k[ok]
        b = blocks.iloc[k]
        q = b["q_start"].to_numpy() + (pos0[idx] - b["start"].to_numpy())
        minus = (b["q_strand"] == "-").to_numpy()
        q = np.where(minus, b["q_size"].to_numpy() - 1 - q, q)
        out_chrom[idx] = b["q_chrom"].to_numpy()
        out_pos[idx] = q + 1
        out_strand[idx] = np.where(minus, "-", "+")
    return pd.DataFrame({"chrom": out_chrom, "pos": pd.array(out_pos, dtype="Int64"), "strand": out_strand})
