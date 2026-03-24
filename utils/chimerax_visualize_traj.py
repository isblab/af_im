import os, time
from chimerax.core.commands import run

def chimerax_pdb_to_movie(
	pdb_file: str,
	output_file_path: str,
	height: int,
	width: int,
	fps: int,
	):
	# Load the PDB file.
	run( session, f"open {pdb_file} coordset true" )

	height, width = 1920, 1080
	fps, quality = 5, "high"
	codec = "apng"

	# Visualization settings.
	run( session, "preset silhouettes" )
	run( session, "color bychain" )
	run( session, "graphics silhouette width 1" )
	run( session, "hide #1" )
	run( session, "show #1 cartoon" )
	# Center and fit the structure
	run( session, "view all" )

	# Get the number of frames.
	models = session.models.list()
	if not models:
		print( "Error: No models loaded" )

	model = models[0]
	num_frames = model.num_coordsets
	print( f"Found {num_frames} frames in the PDB file" )

	# Start movie recording
	print("Recording movie...")
	run( session, f"movie record size {height},{width} supersample 3" )

	# Play through all coordinate sets
	run( session, f"coordset #1 1,{num_frames},1" )
	run( session, f"wait {num_frames}" )

	# Stop recording
	run( session, "movie stop" )

	# Encode movie
	print(f"Encoding movie to {output_file_path}...")

	run( session,
	f"movie encode {output_file_path} format {codec} quality {quality} framerate {fps}" )
	run( session, "close" )


def chimerax_pdb_to_gif(
	pdb_file: str,
	output_file_path: str,
	frame_delay: int,
	loop_flag: int
	):
	# Load the PDB file.
	run( session, f"open {pdb_file} coordset true" )

	# Visualization settings.
	run( session, "preset silhouettes" )
	run( session, "color bychain" )
	run( session, "graphics silhouette width 1" )
	run( session, "hide #1" )
	run( session, "show #1 cartoon" )
	# Center and fit the structure.
	run( session, "view all" )

	# Get the number of frames
	models = session.models.list()
	if not models:
		print( "Error: No models loaded" )

	model = models[0]
	num_frames = model.num_coordsets
	print( f"Found {num_frames} frames in the PDB file" )

	# Temporary file path.
	tmp_dir = os.path.abspath( "./tmp/" )
	os.makedirs( tmp_dir, exist_ok = True )

	# Save each frame as a PNG
	frame_files = []
	for frame in range( 1, num_frames + 1 ):
		# Go to this frame
		run( session, f"coordset #1 {frame}" )
		
		# Save the frame
		frame_file = os.path.join( tmp_dir, f"frame_{frame:04d}.png" )
		run( session, "view all" )
		run( session, f"save {frame_file} supersample 3" )
		frame_files.append( frame_file )
		
		print( f"Saved frame {frame}/{num_frames}" )

	# Convert frames to GIF using ImageMagick
	frame_pattern = os.path.join( tmp_dir, "frame_*.png" )

	print( f"\nConverting {num_frames} frames to GIF..." )
	os.system( f"convert -delay {frame_delay} -loop {loop_flag} {frame_pattern} {output_file_path}" )

	# Clean up temporary frames
	print("Cleaning up temporary files...")
	for frame_file in frame_files:
		os.remove( frame_file )
	os.rmdir( tmp_dir )
	run( session, "close" )


# ChimeraX imports and runs the python scripts in its own namespace and so does not recognize __main__.
if __name__ == "__main__" or __name__.startswith( "ChimeraX_sandbox_" ):
	# chimerax --nogui --script "pdb_to_movie_chimerax.py"
	# 	TODO for later: Problem in encoding in --nogui mode.
	# base_path = "/data2/kartik/IMP_Rewired/imp_dl/benchmark/experiment_modeling/"
	base_path = "/data2/kartik/IMP_Rewired/imp_dl/benchmark/xlmerged_modeling/"
	benchmark_file = "/data2/kartik/IMP_Rewired/imp_dl/benchmark/xlmerged_metadata/selected_xlmerged_benchmark.csv"

	with open( benchmark_file, "r" ) as f:
		pdb_ids = []
		for line in f.readlines():
			line = line.strip().split( "," )
			pdb_ids.append( line[0] )

	make_a = "gif"
	for sys_name in pdb_ids: # , "7xvo", "8wtd", "7r3z", "8sbb"]:
	# for sys_name in ["5gru", "4ut6", "7r3z", "7xae", "7xvo", "7qaj", "7zch", "7yd3", "7yk3", "8c3l",
	# 				"8bzr", "8ct8", "8eho", "8cxj", "8gt0", "8f7d", "8dml", "8grj", "8g9p", "8gzy",
	# 				"8g0q", "8i2f", "8i4g", "8i9q", "8j1n", "8hxq", "8hm3", "8hkh", "8opr", "8pq7",
	# 				"8sbb", "8q6r", "8tft", "8wtd", "7qot", "8weo", "8vyl"]:
		for ver in [2]:
			print( f"Version = {ver}" )
			pdb_file = os.path.join(
				base_path, f"{sys_name}/version_{ver}/{sys_name}_output_models.pdb"
			)
			print( pdb_file )
			if not os.path.exists( pdb_file ):
				print( f"{pdb_file} does not exist..." )
				continue
			if make_a == "movie":
				output_file_path = os.path.join(
					base_path, f"{sys_name}/version_{ver}/{sys_name}_v{ver}.png"
				)
				chimerax_pdb_to_movie(
					pdb_file = pdb_file,
					output_file_path = output_file_path,
					height = 1920,
					width = 1080,
					fps = 5
					)
			elif make_a == "gif":
				output_file_path = os.path.join(
					base_path, f"{sys_name}/version_{ver}/{sys_name}_v{ver}.gif"
				)
				chimerax_pdb_to_gif(
					pdb_file = pdb_file,
					output_file_path = output_file_path,
					frame_delay = 50,  # 1/100th second
					loop_flag = 0 # keep looping the GIF.
					)
			else:
				raise ValueError( f"Incorrect output type specified ({make_a}). Supported movie or gif..." )

