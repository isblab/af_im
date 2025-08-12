#!/bin/bash

sys_name="4rhz"
sys_dir="/data2/kartik/IMP_Rewired/imp_dl/benchmark/rigid_modeling/$sys_name/version_"
tmpdir="./tmp"

for i in {1..7}; do
	pdbfile="${sys_dir}${i}/${sys_name}_output_models.pdb"
	outfile="${sys_dir}${i}/${sys_name}_output_models.gif"
	vmd -dispdev text -e ./utils/vmd_visualize_traj.tcl -args "$pdbfile" "$outfile" "$tmpdir" 0.9 10
done