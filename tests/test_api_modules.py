"""
Contains test cases for the following modules:
	SeqResDict
"""
from typing import Dict
import numpy as np
from data.api_data_modules import SeqResDict



class TestSeqResDict():
	"""
	Test for SeqResDict module.
	1d5t.cif has been modified to test the following cases:
		Handling missing residues
		Handling artificial/engineered residues
	"""
	def __init__( self ):
		self.test_pdb_id = ["3c09"]
		self.test_struct_dir = "./"



	def forward( self ):
		"""
		"""
		self.check_results()


	def run_seqres_dict_module( self ):
		"""
		"""
		obj = SeqResDict(
			pdb_ids_list = [self.test_pdb_id],
			pdb_struct_dir = self.test_struct_dir,
			cores = 1,
			max_sys_length = 1400,
			frac_coverage = 0.99
			)
		( entry_id,
			cif_dict,
			coverage,
			logs ) = obj.get_cif_dict_for_entry( self.test_pdb_id[0] )
		return cif_dict, coverage


	def test_output( self ):
		"""
		Desired output dict for input PDB ID.
		"""
		output_cif_dict = {
			"1": {
				"L":{
					"seq": "TQSPS",
					"start_pos": 5,
					"end_pos": 9,
					"res_num": np.array( [5, 6, 7, 8, 9] ),
					"start_seq_id": 5,
					"end_seq_id": 9,
					"seq_id": np.array( [5, 6, 7, 8, 9] )
				}
			},
			"2": {
				"H":{
					"seq": "VQLVQSG",
					"start_pos": 2,
					"end_pos": 8,
					"res_num": np.array( [2, 3, 6, 7, 8] ),
					"start_seq_id": 2,
					"end_seq_id": 8,
					"seq_id": np.array( [2, 3, 6, 7, 8] )
				}
			},
			"3": {
				"A":{
					"seq": "KKVCN",
					"start_pos": 310,
					"end_pos": 314,
					"res_num": np.array( [310, 311, 312, 313, 314] ),
					"start_seq_id": 4,
					"end_seq_id": 8,
					"seq_id": np.array( [4, 5, 6, 7, 8] )
				}
			}
		}
		coverage = [5/5, 5/7, 5/5]
		return output_cif_dict, coverage


	def check_results( self ):
		"""
		"""
		cif_dict, coverage = self.run_seqres_dict_module()
		output_cif_dict, out_coverage = self.test_output()

		print( cif_dict )
		passed = []
		out_entity_ids = list( output_cif_dict.keys() )
		entry_ids = list( cif_dict.keys() )
		if all( [a == b for a, b in zip( out_entity_ids, entry_ids )] ):
			print( "Test passed. Correct entity_id's selected..." )
			passed.append( True )
		else:
			passed.append( False )
			raise ValueError( "Test failed. Incorrect entity_id's..." )

		for entity_id in output_cif_dict:
			for out_chain_id, chain_id in zip(
				output_cif_dict[entity_id],
				cif_dict[entity_id] ):
				if out_chain_id != chain_id:
					passed.append( False )
					raise ValueError( f"Test failed. Incorrect chain_id {chain_id} -- {out_chain_id}..." )

				out_seq = output_cif_dict[entity_id][out_chain_id]["seq"]
				seq = cif_dict[entity_id][out_chain_id]["seq"]
				if out_seq != seq:
					passed.append( False )
					raise ValueError( f"Test failed. Entity; {entity_id}  Chain: {out_chain_id}" +
									f" --> Incorrect sequence {seq} -- {out_seq}..." )

				out_start = output_cif_dict[entity_id][out_chain_id]["start_seq_id"]
				start = cif_dict[entity_id][out_chain_id]["start_seq_id"]
				if out_start != start:
					passed.append( False )
					raise ValueError( f"Test failed. Entity; {entity_id}  Chain: {out_chain_id}" +
									f" --> Incorrect start residue {start} -- {out_start}..." )

				out_end = output_cif_dict[entity_id][out_chain_id]["end_seq_id"]
				end = cif_dict[entity_id][out_chain_id]["end_seq_id"]
				if out_end != end:
					passed.append( False )
					raise ValueError( f"Test failed. Entity; {entity_id}  Chain: {out_chain_id}" +
									f" --> Incorrect end residue {end} -- {out_end}..." )

				out_res_num = output_cif_dict[entity_id][out_chain_id]["res_num"]
				res_num = cif_dict[entity_id][out_chain_id]["res_num"]
				if any( [a != b for a,b in zip( out_res_num, res_num )] ):
					raise ValueError( f"Test failed. Entity; {entity_id}  Chain: {out_chain_id}" +
										f" --> Incorrect res_num {res_num} -- {out_res_num}...")

		if any( [a != b for a,b in zip( out_coverage, coverage )] ):
			passed.append( False )
			raise ValueError( f"Test failed. Incorrect coverage computed: {coverage} -- {out_coverage}" )
		else:
			passed.append( True )

		print( "All tests passed" )


if __name__ == "__main__":
	TestSeqResDict().forward()