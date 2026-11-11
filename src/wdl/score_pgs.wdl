version 1.0

# Score a polygenic score with plink2 --score, one task per chromosome, then sum across chromosomes.
# The final scores and summary are copied to output_dir.
#
# score_file: ID A1 BETA with a header (e.g. the score_file output of prscs.wdl). IDs must match
# the variant IDs in the genotype files.
#
# chrom_files: one inner array per chromosome holding that chromosome's files, recognized by
# extension:
#   format "pgen": [.pgen, .pvar or .pvar.zst, .psam]
#   format "bed":  [.bed, .bim, .fam]
#   format "bgen": [.bgen, .sample]
#   format "vcf":  [.vcf.gz / .vcf / .bcf]  (dosages read from the vcf_dosage FORMAT field)

workflow score_pgs {
  input {
    File score_file
    Array[Array[File]] chrom_files
    String format
    String bgen_ref = "ref-first"
    String vcf_dosage = "DS"
    # Optional: only score these samples (plink2 --keep file: IID, or FID IID)
    File? keep
    String out_prefix = "pgs"
    # gs:// folder the final outputs are copied to, e.g. gs://<bucket>/working/PGS
    String output_dir

    String plink2_docker = "pgscatalog/plink2:2.00a5.10"
    String gcloud_docker = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"
    Int cpu = 4
    Int mem_gb = 16
    Int preemptible = 1
  }

  scatter (files in chrom_files) {
    call score_chrom {
      input:
        score_file = score_file,
        files = files,
        format = format,
        bgen_ref = bgen_ref,
        vcf_dosage = vcf_dosage,
        keep = keep,
        docker = plink2_docker,
        cpu = cpu,
        mem_gb = mem_gb,
        preemptible = preemptible
    }
  }

  call combine {
    input:
      sscores = score_chrom.sscore,
      used = score_chrom.used,
      out_prefix = out_prefix,
      docker = plink2_docker,
      preemptible = preemptible
  }

  call copy_outputs {
    input:
      files = [combine.scores, combine.summary],
      output_dir = output_dir,
      docker = gcloud_docker,
      preemptible = preemptible
  }

  output {
    # IID ALLELE_CT NAMED_ALLELE_DOSAGE_SUM PGS (PGS = sum of BETA x dosage over all chromosomes)
    File scores = combine.scores
    # Variants used per chromosome, labeled by genotype file name
    File summary = combine.summary
    Array[File] logs = score_chrom.log
  }
}

task score_chrom {
  input {
    File score_file
    Array[File] files
    String format
    String bgen_ref
    String vcf_dosage
    File? keep
    String docker
    Int cpu
    Int mem_gb
    Int preemptible
  }

  Int disk_gb = ceil(size(files, "GB") * 1.2 + size(score_file, "GB")) + 20

  command <<<
    set -euo pipefail
    geno= vars= samples=
    for f in ~{sep=" " files}; do
      case "$f" in
        *.pgen|*.bed|*.bgen|*.vcf|*.vcf.gz|*.vcf.bgz|*.bcf) geno="$f" ;;
        *.pvar|*.pvar.zst|*.bim) vars="$f" ;;
        *.psam|*.fam|*.sample) samples="$f" ;;
        *.tbi|*.csi|*.bgi) ;;  # indexes: localized next to the data, not passed to plink2
        *) echo "Don't know what to do with $f" >&2; exit 1 ;;
      esac
    done

    case "~{format}" in
      pgen) src="--pgen $geno --pvar $vars --psam $samples" ;;
      bed)  src="--bed $geno --bim $vars --fam $samples" ;;
      bgen) src="--bgen $geno ~{bgen_ref} --sample $samples" ;;
      vcf)  case "$geno" in
              *.bcf) src="--bcf $geno dosage=~{vcf_dosage}" ;;
              *)     src="--vcf $geno dosage=~{vcf_dosage}" ;;
            esac ;;
      *) echo "format must be pgen, bed, bgen or vcf (got ~{format})" >&2; exit 1 ;;
    esac

    # e.g. acaf_threshold.chr1.maf01.geno02.eur.bed -> acaf_threshold.chr1.maf01.geno02.eur
    name=$(basename "$geno"); name=${name%.gz}; name=${name%.bgz}; name=${name%.*}
    plink2 $src \
      ~{"--keep " + keep} \
      --score "~{score_file}" 1 2 3 header-read cols=+scoresums list-variants \
      --threads ~{cpu} --memory ~{mem_gb * 900} \
      --out "$name"

    mv "$name.sscore" chrom.sscore
    mv "$name.log" "$name.score.log"
    printf '%s\t%s\n' "$name" "$(wc -l < "$name.sscore.vars")" > used.tsv
  >>>

  output {
    File sscore = "chrom.sscore"
    File used = "used.tsv"
    File log = glob("*.score.log")[0]
  }

  runtime {
    docker: docker
    cpu: cpu
    memory: mem_gb + " GB"
    disks: "local-disk " + disk_gb + " SSD"
    preemptible: preemptible
  }
}

task combine {
  input {
    Array[File] sscores
    Array[File] used
    String out_prefix
    String docker
    Int preemptible
  }

  command <<<
    set -euo pipefail
    # Sum allele counts, dosage sums and score sums by IID across the per-chromosome .sscore files
    awk -F'\t' -v OFS='\t' '
      FNR == 1 {
        delete col
        for (i = 1; i <= NF; i++) { h = $i; sub(/^#/, "", h); col[h] = i }
        for (h in col) if (h ~ /_SUM$/ && h != "NAMED_ALLELE_DOSAGE_SUM") sc = col[h]
        next
      }
      {
        id = $(col["IID"])
        if (!(id in pgs)) order[++n] = id
        ct[id] += $(col["ALLELE_CT"]); ds[id] += $(col["NAMED_ALLELE_DOSAGE_SUM"]); pgs[id] += $sc
      }
      END {
        print "IID", "ALLELE_CT", "NAMED_ALLELE_DOSAGE_SUM", "PGS"
        for (i = 1; i <= n; i++) { id = order[i]; print id, ct[id], ds[id], pgs[id] }
      }' ~{sep=" " sscores} > "~{out_prefix}.sscore.tsv"

    printf 'FILE\tVARIANTS_USED\n' > "~{out_prefix}.score_summary.tsv"
    cat ~{sep=" " used} >> "~{out_prefix}.score_summary.tsv"
    cat "~{out_prefix}.score_summary.tsv"
  >>>

  output {
    File scores = "~{out_prefix}.sscore.tsv"
    File summary = "~{out_prefix}.score_summary.tsv"
  }

  runtime {
    docker: docker
    cpu: 1
    memory: "8 GB"
    disks: "local-disk 20 SSD"
    preemptible: preemptible
  }
}

task copy_outputs {
  input {
    Array[File] files
    String output_dir
    String docker
    Int preemptible
  }

  String dest = sub(output_dir, "/+$", "")

  command <<<
    set -euo pipefail
    gcloud storage cp ~{sep=" " files} "~{dest}/"
  >>>

  runtime {
    docker: docker
    cpu: 1
    memory: "2 GB"
    disks: "local-disk 20 SSD"
    preemptible: preemptible
  }
}
