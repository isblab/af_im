#!/bin/bash

for sys_name in "4rhz" "8wtd"; do
	sys_dir="/data2/kartik/IMP_Rewired/imp_dl/benchmark/rigid_modeling/$sys_name/version_"
	tmpdir="./tmp"
	for sv in $(seq 0 0.1 0.5); do
		# if [ "$sv" -eq 0 ]; then
		# # if (( $(echo "$sv == 0" | bc -l) )); then
		# 	s=$((1+$sv))
		# 	e=$((10+$sv))
		# else
		s=$(echo "1+$sv"| bc)
		e=$(echo "10+$sv"| bc)
	    # Remove trailing 0 after decimal.
	    s=${s%\.0}
	    e=${e%\.0}
		# fi
		for i in $(seq $s 1 $e); do
			echo "$sys_name $sv $s $e $i"
			pdbfile="${sys_dir}${i}/${sys_name}_output_models.pdb"
			outfile="${sys_dir}${i}/${sys_name}_v${i}.gif"
			vmd -dispdev text -e ./utils/vmd_visualize_traj.tcl -args "$pdbfile" "$outfile" "$tmpdir" 0.9 10
		done
	done
done
