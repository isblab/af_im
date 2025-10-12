#!/bin/bash

for sys_name in "4rhz" "8wtd"; do
	sys_dir="/data2/kartik/IMP_Rewired/imp_dl/benchmark/experiment_modeling/$sys_name/version_"
	tmpdir="./tmp"
	# for i in $(seq 0 1 2); do
	# 	# Remove trailing 0 after decimal.
	# 	i=${i%\.0}
	# 	echo "$sys_name $v $s $e $i"
	# 	pdbfile="${sys_dir}${i}/${sys_name}_output_models.pdb"
	# 	outfile="${sys_dir}${i}/${sys_name}_v${i}.gif"
	# 	vmd -dispdev text -e ./utils/vmd_visualize_traj.tcl -args "$pdbfile" "$outfile" "$tmpdir" 0.9 10
	# done

	for v in $(seq 4); do
		s=$(echo "$v+0"| bc)
		e=$(echo "$v+0.1"| bc)
		for i in $(seq $s 0.1 $e); do
			# Remove trailing 0 after decimal.
			i=${i%\.0}
			echo "$sys_name $v $s $e $i"
			pdbfile="${sys_dir}${i}/${sys_name}_output_models.pdb"
			outfile="${sys_dir}${i}/${sys_name}_v${i}.gif"
			vmd -dispdev text -e ./utils/vmd_visualize_traj.tcl -args "$pdbfile" "$outfile" "$tmpdir" 1.0 20
			done
	done
done
