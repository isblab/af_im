#!/bin/bash

fasta=$1
output_dir=$2
mode=$3
alignments_dir=$4

tic=$(date +%s)


if [ "$mode" = "mono" ]; then
    echo "Running OpenFold monomer prediction"
    if [ "$alignments_dir" = "None" ]; then
        echo "Creating alignments..."
        python3 run_pretrained_openfold.py \
                "$fasta" \
                /data/alpha-fold-db/pdb_mmcif/mmcif_files/ \
                --uniref90_database_path /data/alpha-fold-db/uniref90/uniref90.fasta \
                --mgnify_database_path /data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa \
                --pdb70_database_path /data/alpha-fold-db/pdb70/pdb70 \
                --uniclust30_database_path /data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
                --bfd_database_path /data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --jackhmmer_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/jackhmmer \
                --hhblits_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/hhblits \
                --hhsearch_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/hhsearch \
                --kalign_binary_path lib/conda/envs/openfold_env/bin/kalign \
                --config_preset "model_1_ptm" \
                --model_device "cuda:0" \
                --output_dir "$output_dir" \
                --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt
    else
        echo "Using pre-computed alignments..."
        python3 run_pretrained_openfold.py \
                "$fasta" \
                /data/alpha-fold-db/pdb_mmcif/mmcif_files/ \
                --use_precomputed_alignments "$alignment_dir" \
                --uniref90_database_path /data/alpha-fold-db/uniref90/uniref90.fasta \
                --mgnify_database_path /data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa \
                --pdb70_database_path /data/alpha-fold-db/pdb70/pdb70 \
                --uniclust30_database_path /data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
                --bfd_database_path /data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --jackhmmer_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/jackhmmer \
                --hhblits_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/hhblits \
                --hhsearch_binary_path /home/greenie/miniforge3/envs/openfold_env/bin/hhsearch \
                --kalign_binary_path lib/conda/envs/openfold_env/bin/kalign \
                --config_preset "model_1_ptm" \
                --model_device "cuda:0" \
                --output_dir "$output_dir" \
                --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt
    fi

elif [ "$mode" = "multi" ]; then
    echo "Running OpenFold multimer prediction"
    if [ "$alignments_dir" = "None" ]; then
        echo "Creating alignments..."
        python3 run_pretrained_openfold.py \
                "$fasta" \
                /data/alpha-fold-db/pdb_mmcif/mmcif_files/ \
                --uniref90_database_path /data/alpha-fold-db/uniref90/uniref90.fasta \
                --mgnify_database_path /data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa \
                --pdb_seqres_database_path /data/alpha-fold-db/pdb_seqres/pdb_seqres.txt \
                --uniref30_database_path /data/alpha-fold-db/uniref30/UniRef30_2021_03 \
                --uniprot_database_path /data/alpha-fold-db/uniprot/uniprot.fasta \
                --bfd_database_path /data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --jackhmmer_binary_path /home/user/miniforge3/envs/openfold_env/bin/jackhmmer \
                --hhblits_binary_path /home/user/miniforge3/envs/openfold_env/bin/hhblits \
                --hmmsearch_binary_path /home/user/miniforge3/envs/openfold_env/bin/hmmsearch \
                --hmmbuild_binary_path /home/user/miniforge3/envs/openfold_env/bin/hmmbuild \
                --kalign_binary_path /home/user/miniforge3/envs/openfold_env/bin/kalign \
                --config_preset "model_1_multimer_v3" \
                --model_device "cuda:0" \
                --output_dir "$output_dir"
    else
        echo "Using pre-computed alignments..."
        python3 run_pretrained_openfold.py \
                "$fasta" \
                /data/alpha-fold-db/pdb_mmcif/mmcif_files/ \
                --use_precomputed_alignments "$alignments_dir" \
                --uniref90_database_path /data/alpha-fold-db/uniref90/uniref90.fasta \
                --mgnify_database_path /data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa \
                --pdb_seqres_database_path /data/alpha-fold-db/pdb_seqres/pdb_seqres.txt \
                --uniref30_database_path /data/alpha-fold-db/uniref30/UniRef30_2021_03 \
                --uniprot_database_path /data/alpha-fold-db/uniprot/uniprot.fasta \
                --bfd_database_path /data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --jackhmmer_binary_path /home/user/miniforge3/envs/openfold_env/bin/jackhmmer \
                --hhblits_binary_path /home/user/miniforge3/envs/openfold_env/bin/hhblits \
                --hmmsearch_binary_path /home/user/miniforge3/envs/openfold_env/bin/hmmsearch \
                --hmmbuild_binary_path /home/user/miniforge3/envs/openfold_env/bin/hmmbuild \
                --kalign_binary_path /home/user/miniforge3/envs/openfold_env/bin/kalign \
                --config_preset "model_1_multimer_v3" \
                --model_device "cuda:0" \
                --output_dir "$output_dir"

    fi

else
        echo "Incorrect mode. (Should be either mono or multi)"
fi

toc=$(date +%s)
echo "$tic   $toc"
total_time=$(($toc - $tic))
echo "Total time taken (sec): $total_time"
echo "Total time taken (hours): $(( total_time / 3600 ))"