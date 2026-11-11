version 1.0

# PRS-CS posterior SNP weights for scoring a GRCh38 target dataset (e.g. AoU imputed array data).
#
# PRS-CS works in rsIDs and its LD reference (snpinfo_*_hm3) uses GRCh37 positions, while the
# target data is GRCh38 and its variant IDs are usually chr:pos:ref:alt. So, per chromosome:
#   1. match_target: find the HapMap3 SNPs in the target's variant list by GRCh38 position and
#      alleles (hm3 = snpinfo lifted to GRCh38 with aou_utils.array.pgs.lift_snpinfo). Writes a
#      .bim of the matched SNPs named by rsID (what PRS-CS reads as the target SNP list) and an
#      rsID -> target variant ID map.
#   2. prscs: run PRS-CS on that chromosome.
# Then make_score_file joins the weights to the target IDs, giving a plink2 --score file
# (ID, A1, BETA) for score_pgs.wdl.
#
# Inputs ordered by chromosome: ld_blocks and target_variants must both be chr1..chr22 in order.

workflow prscs {
  input {
    # GWAS summary statistics in PRS-CS format: SNP A1 A2 BETA|OR SE|P (aou_utils.array.pgs.format_sumstats)
    File sumstats
    Int n_gwas
    # snpinfo_ukbb_hm3 (or snpinfo_1kg_hm3) from the LD reference folder
    File ld_snpinfo
    # ldblk_ukbb_chr1.hdf5 ... ldblk_ukbb_chr22.hdf5, in chromosome order
    Array[File] ld_blocks
    # Folder name PRS-CS expects; must contain 'ukbb' or '1kg' and match the snpinfo/hdf5 names
    String ld_name = "ldblk_ukbb_eur"
    # HapMap3 SNPs with GRCh38 positions: CHR SNP BP A1 A2 (aou_utils.array.pgs.lift_snpinfo)
    File hm3_grch38
    # One variant list per chromosome, chr1..chr22 in order: .pvar, .pvar.zst, .bim, or a VCF/BCF
    Array[File] target_variants

    # PRS-CS settings; phi unset = learn phi from the data (fine for large GWAS)
    Float? phi
    Int n_iter = 1000
    Int n_burnin = 500
    Int thin = 5
    Int seed = 1

    String out_prefix = "prscs"
    String prscs_docker = "quay.io/biocontainers/prscs:1.1.0--hdfd78af_0"
    String plink2_docker = "pgscatalog/plink2:2.00a5.10"
    Int prscs_cpu = 4
    Int prscs_mem_gb = 16
    Int preemptible = 1
  }

  scatter (i in range(length(ld_blocks))) {
    Int chrom = i + 1

    call match_target {
      input:
        variants = target_variants[i],
        hm3 = hm3_grch38,
        chrom = chrom,
        docker = plink2_docker,
        preemptible = preemptible
    }

    call run_prscs {
      input:
        sumstats = sumstats,
        n_gwas = n_gwas,
        bim = match_target.bim,
        snpinfo = ld_snpinfo,
        ld_block = ld_blocks[i],
        ld_name = ld_name,
        chrom = chrom,
        phi = phi,
        n_iter = n_iter,
        n_burnin = n_burnin,
        thin = thin,
        seed = seed,
        docker = prscs_docker,
        cpu = prscs_cpu,
        mem_gb = prscs_mem_gb,
        preemptible = preemptible
    }
  }

  call make_score_file {
    input:
      weights = run_prscs.weights,
      idmaps = match_target.idmap,
      match_summaries = match_target.summary,
      out_prefix = out_prefix,
      docker = plink2_docker,
      preemptible = preemptible
  }

  output {
    # CHR SNP BP_GRCh37 A1 A2 BETA ID: PRS-CS weights with the matching target variant ID
    File weights = make_score_file.weights_with_ids
    # ID A1 BETA, for plink2 --score (score_pgs.wdl)
    File score_file = make_score_file.score_file
    # Per chromosome: HapMap3 SNPs on the chromosome, matched in target, PRS-CS weights written
    File summary = make_score_file.summary
    Array[File] prscs_logs = run_prscs.log
  }
}

task match_target {
  input {
    File variants
    File hm3
    Int chrom
    String docker
    Int preemptible
  }

  Int disk_gb = ceil(size(variants, "GB") * 2 + size(hm3, "GB")) + 10

  command <<<
    set -euo pipefail
    v="~{variants}"
    case "$v" in
      *.vcf|*.vcf.gz|*.vcf.bgz) src="--vcf $v" ;;
      *.bcf) src="--bcf $v" ;;
      *) src="--pvar $v" ;;  # .pvar, .pvar.zst, or .bim
    esac

    # HapMap3 positions on this chromosome as 1-based ranges
    awk -F'\t' -v c=~{chrom} 'NR > 1 && $1 == c {print $1, $3, $3, $2}' OFS='\t' ~{hm3} > hm3.ranges

    # --chr errors out if the file has no variants on this chromosome (inputs out of order)
    plink2 $src --chr ~{chrom} --extract bed1 hm3.ranges --output-chr 26 \
      --make-just-pvar --out candidates

    # Keep target variants whose alleles match the HapMap3 SNP at that position. The .bim uses
    # A1 = target ALT, A2 = target REF, so PRS-CS weights come out per target ALT allele.
    # Skip variant IDs that are missing ('.') or duplicated, and rsIDs matched twice.
    awk -F'\t' -v OFS='\t' -v c=~{chrom} '
      FNR == NR {
        if (FNR > 1 && $1 == c) { k = $3; if (!(k in rs)) { rs[k] = $2; a1[k] = $4; a2[k] = $5 } }
        next
      }
      /^##/ { next }
      /^#CHROM/ { for (i = 1; i <= NF; i++) col[$i] = i; next }
      {
        k = $(col["POS"]); if (!(k in rs)) next
        ref = $(col["REF"]); alt = $(col["ALT"]); id = $(col["ID"])
        if (!((ref == a1[k] && alt == a2[k]) || (ref == a2[k] && alt == a1[k]))) next
        n++; pos[n] = k; vid[n] = id; vref[n] = ref; valt[n] = alt; idct[id]++
      }
      END {
        for (i = 1; i <= n; i++) {
          s = rs[pos[i]]
          if (vid[i] == "." || idct[vid[i]] > 1 || (s in used)) { skipped++; continue }
          used[s] = 1; m++
          print c, s, 0, pos[i], valt[i], vref[i] > "target.bim"
          print s, vid[i] > "idmap.tsv"
        }
        nhm3 = 0; for (k in rs) nhm3++
        print c, nhm3, m + 0, skipped + 0 > "summary.tsv"
      }' ~{hm3} candidates.pvar

    touch target.bim idmap.tsv
    cat summary.tsv
  >>>

  output {
    File bim = "target.bim"
    File idmap = "idmap.tsv"
    File summary = "summary.tsv"
  }

  runtime {
    docker: docker
    cpu: 2
    memory: "8 GB"
    disks: "local-disk " + disk_gb + " SSD"
    preemptible: preemptible
  }
}

task run_prscs {
  input {
    File sumstats
    Int n_gwas
    File bim
    File snpinfo
    File ld_block
    String ld_name
    Int chrom
    Float? phi
    Int n_iter
    Int n_burnin
    Int thin
    Int seed
    String docker
    Int cpu
    Int mem_gb
    Int preemptible
  }

  Int disk_gb = ceil(size(ld_block, "GB") + size(snpinfo, "GB") + size(sumstats, "GB")) + 10

  command <<<
    set -euo pipefail
    export MKL_NUM_THREADS=~{cpu} NUMEXPR_NUM_THREADS=~{cpu} OMP_NUM_THREADS=~{cpu} OPENBLAS_NUM_THREADS=~{cpu}

    blk=$(basename "~{ld_block}")
    case "$blk" in
      *_chr~{chrom}.hdf5) ;;
      *) echo "LD block $blk isn't chromosome ~{chrom}: pass ld_blocks in order chr1..chr22" >&2; exit 1 ;;
    esac

    # PRS-CS wants one folder holding snpinfo_* and ldblk_*_chrN.hdf5
    mkdir -p "ref/~{ld_name}" out
    ln -s "~{snpinfo}" "ref/~{ld_name}/$(basename ~{snpinfo})"
    ln -s "~{ld_block}" "ref/~{ld_name}/$blk"
    cp "~{bim}" target.bim

    PRScs.py \
      --ref_dir="ref/~{ld_name}" \
      --bim_prefix=target \
      --sst_file="~{sumstats}" \
      --n_gwas=~{n_gwas} \
      --out_dir=out/prscs \
      --chrom=~{chrom} \
      --n_iter=~{n_iter} --n_burnin=~{n_burnin} --thin=~{thin} \
      --seed=~{seed} \
      ~{"--phi=" + phi} 2>&1 | tee prscs_chr~{chrom}.log

    cat out/prscs_pst_eff_*_chr~{chrom}.txt > weights_chr~{chrom}.txt
  >>>

  output {
    File weights = "weights_chr~{chrom}.txt"
    File log = "prscs_chr~{chrom}.log"
  }

  runtime {
    docker: docker
    cpu: cpu
    memory: mem_gb + " GB"
    disks: "local-disk " + disk_gb + " SSD"
    preemptible: preemptible
  }
}

task make_score_file {
  input {
    Array[File] weights
    Array[File] idmaps
    Array[File] match_summaries
    String out_prefix
    String docker
    Int preemptible
  }

  command <<<
    set -euo pipefail
    cat ~{sep=" " idmaps} > idmap.tsv
    cat ~{sep=" " weights} > weights.tsv

    # PRS-CS output: CHR SNP BP(GRCh37) A1 A2 BETA, with A1 = target ALT (from target.bim)
    awk -F'\t' -v OFS='\t' -v w="~{out_prefix}.weights.tsv" -v s="~{out_prefix}.score.tsv" '
      BEGIN { print "CHR", "SNP", "BP_GRCh37", "A1", "A2", "BETA", "ID" > w; print "ID", "A1", "BETA" > s }
      FNR == NR { id[$1] = $2; next }
      ($2 in id) { print $1, $2, $3, $4, $5, $6, id[$2] > w; print id[$2], $4, $6 > s; nw[$1]++ }
      END { for (c in nw) print c, nw[c] > "weights_per_chrom.tsv" }
    ' idmap.tsv weights.tsv
    touch weights_per_chrom.tsv

    # summary: chromosome, HapMap3 SNPs, matched in target, skipped (bad/duplicate ID), weights
    printf 'CHR\tHM3_SNPS\tMATCHED\tSKIPPED\tWEIGHTS\n' > "~{out_prefix}.summary.tsv"
    cat ~{sep=" " match_summaries} | sort -k1,1n \
      | awk -v OFS='\t' 'FNR == NR { nw[$1] = $2; next } { print $1, $2, $3, $4, nw[$1] + 0 }' \
          weights_per_chrom.tsv - >> "~{out_prefix}.summary.tsv"
    cat "~{out_prefix}.summary.tsv"
  >>>

  output {
    File weights_with_ids = "~{out_prefix}.weights.tsv"
    File score_file = "~{out_prefix}.score.tsv"
    File summary = "~{out_prefix}.summary.tsv"
  }

  runtime {
    docker: docker
    cpu: 1
    memory: "4 GB"
    disks: "local-disk 20 SSD"
    preemptible: preemptible
  }
}
