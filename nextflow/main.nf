#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * IRIS: Isoform-Resolved Inference for Single-cells
 *
 * Isoform-aware single-cell analysis for long-read sequencing data.
 * Accepts pre-processed inputs from any upstream pipeline
 * (Cell Ranger, STARsolo, wf-single-cell, FLAMES, etc.)
 */


// --- Processes ---

process cluster_analysis {
    cpus 2
    memory { 8.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "gene_matrix"
    output:
        path "clusters.tsv", emit: clusters
        path "cluster.umap.tsv", emit: umap
        path "marker_genes.tsv", emit: markers
        path "cluster_summary.json", emit: summary
    script:
    """
    iris cluster gene_matrix \
        --cluster_method ${params.cluster_method} \
        --resolution ${params.cluster_resolution}
    """
}

process dual_layer_clustering {
    cpus 4
    memory { 16.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "gene_matrix"
        path "transcript_matrix"
        path "gene_transcript_map.tsv"
    output:
        path "gene_clusters.tsv", emit: gene_clusters
        path "isoform_clusters.tsv", emit: isoform_clusters
        path "joint_clusters.tsv", emit: joint_clusters
        path "joint.umap.tsv", emit: joint_umap
        path "isoform_diversity.tsv", emit: diversity
        path "cluster_comparison.json", emit: comparison
    script:
    """
    iris dual-cluster gene_matrix transcript_matrix \
        --gene_transcript_map gene_transcript_map.tsv \
        --cluster_method ${params.cluster_method} \
        --resolution ${params.cluster_resolution}
    """
}

process cell_type_annotation {
    cpus 2
    memory { 8.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "gene_matrix"
        path "clusters.tsv"
    output:
        path "cell_type_annotations.tsv", emit: annotations
        path "cluster_cell_types.tsv", emit: cluster_types
        path "cell_type_summary.json", emit: summary
    script:
    """
    iris annotate gene_matrix \
        --clusters clusters.tsv \
        --species ${params.species}
    """
}

process differential_transcript_usage {
    cpus 4
    memory { 16.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "transcript_matrix"
        path "clusters.tsv"
        path "gene_transcript_map.tsv"
    output:
        path "dtu_results.tsv", emit: dtu_results
        path "isoform_switching.tsv", emit: switching
        path "dtu_summary.json", emit: summary
    script:
    """
    iris dtu transcript_matrix \
        --clusters clusters.tsv \
        --gene_transcript_map gene_transcript_map.tsv \
        --test_method ${params.dtu_test_method}
    """
}

process novel_isoform_discovery {
    cpus 2
    memory { 16.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "transcript_matrix"
        path "clusters.tsv"
        path "gene_transcript_map.tsv"
        path "annotated_gtfs/*"
    output:
        path "novel_isoform_catalog.tsv", emit: catalog
        path "novel_isoform_enrichment.tsv", emit: enrichment
        path "novel_isoform_summary.json", emit: summary
    script:
    """
    iris novel-isoforms transcript_matrix \
        --annotated_gtfs annotated_gtfs/* \
        --clusters clusters.tsv \
        --gene_transcript_map gene_transcript_map.tsv
    """
}

process isoform_trajectory {
    cpus 4
    memory { 16.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "gene_matrix"
        path "transcript_matrix"
        path "gene_transcript_map.tsv"
    output:
        path "pseudotime.tsv", emit: pseudotime
        path "isoform_dynamics.tsv", emit: dynamics
        path "trajectory_switching.tsv", emit: switching
        path "trajectory_summary.json", emit: summary
    script:
    """
    iris trajectory gene_matrix \
        --transcript_matrix_dir transcript_matrix \
        --gene_transcript_map gene_transcript_map.tsv
    """
}

process allele_specific_expression {
    cpus 4
    memory { 32.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "tagged.bam"
        path "tagged.bam.bai"
        path "clusters.tsv"
    output:
        path "ase_results.tsv", emit: results
        path "ase_summary.json", emit: summary
    script:
    """
    iris ase tagged.bam --clusters clusters.tsv
    """
}

process export_anndata {
    cpus 1
    memory { 16.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "gene_matrix"
        path "transcript_matrix"
        path inputs, stageAs: "inputs/*"
    output:
        path "iris.h5ad", emit: h5ad
    script:
    """
    iris export gene_matrix \
        --transcript_matrix_dir transcript_matrix \
        --output iris.h5ad \
        \$(ls inputs/joint_clusters.tsv 2>/dev/null && echo "--joint_clusters inputs/joint_clusters.tsv") \
        \$(ls inputs/joint.umap.tsv 2>/dev/null && echo "--joint_umap inputs/joint.umap.tsv") \
        \$(ls inputs/cell_type_annotations.tsv 2>/dev/null && echo "--cell_type_annotations inputs/cell_type_annotations.tsv") \
        \$(ls inputs/isoform_diversity.tsv 2>/dev/null && echo "--isoform_diversity inputs/isoform_diversity.tsv") \
        \$(ls inputs/dtu_results.tsv 2>/dev/null && echo "--dtu_results inputs/dtu_results.tsv")
    """
}

process generate_report {
    cpus 1
    memory { 4.GB * task.attempt }
    errorStrategy { task.exitStatus in [137, 140] ? 'retry' : 'terminate' }
    maxRetries 2
    publishDir "${params.out_dir}/${params.sample_id}", mode: 'copy'
    input:
        path "results"
    output:
        path "iris_report.html", emit: report
    script:
    """
    iris report --out_dir results --output iris_report.html
    """
}


// --- Workflow ---

workflow {
    // Validate required inputs
    if (!params.gene_matrix_dir || !params.transcript_matrix_dir) {
        error "Required: --gene_matrix_dir and --transcript_matrix_dir"
    }

    gene_matrix = Channel.fromPath(params.gene_matrix_dir, type: 'dir')
    transcript_matrix = Channel.fromPath(params.transcript_matrix_dir, type: 'dir')
    gene_tx_map = Channel.fromPath(params.gene_transcript_map)

    // 1. Dual-layer clustering
    dual_layer_clustering(gene_matrix, transcript_matrix, gene_tx_map)

    // 2. Cell type annotation
    if (params.cell_type_annotation) {
        cell_type_annotation(
            gene_matrix,
            dual_layer_clustering.out.joint_clusters)
    }

    // 3. DTU
    if (params.differential_transcript_usage) {
        differential_transcript_usage(
            transcript_matrix,
            dual_layer_clustering.out.joint_clusters,
            gene_tx_map)
    }

    // 4. Novel isoforms (if GTFs provided)
    if (params.novel_isoform_discovery && params.annotated_gtfs) {
        novel_isoform_discovery(
            transcript_matrix,
            dual_layer_clustering.out.joint_clusters,
            gene_tx_map,
            Channel.fromPath(params.annotated_gtfs))
    }

    // 5. Trajectory
    if (params.isoform_trajectory) {
        isoform_trajectory(gene_matrix, transcript_matrix, gene_tx_map)
    }

    // 6. ASE (if BAM provided)
    if (params.allele_specific_expression && params.tagged_bam) {
        bam = Channel.fromPath(params.tagged_bam)
        bai = Channel.fromPath("${params.tagged_bam}.bai")
        allele_specific_expression(
            bam, bai,
            dual_layer_clustering.out.joint_clusters)
    }
}
