"""
This script runs some test cases to sanity check data_gathering.py.
I have created dummy inputs for the test:
	sys_conf_dummy0.json
	sys_conf_dummy1.json
"""
from typing import List, Dict
import os
import json
import copy
import numpy as np
import pandas as pd
import ml_collections as mlc

from data_gathering import DataGathering
from utils.utils import ( open_file_handler, read_json )


##############################################################################
#----------------------------------------------------------------------------#
##############################################################################
class TestDataGathering():
	"""
	Runs some test cases to sanity check functions
		in the DataGathering class.
	"""
	def __init__( self ):
		self.dummy_conf_file = os.path.join( 
									os.path.abspath( "./tests/sys_conf_dummy" )
									)


	def forward( self ):
		"""
		Load the dummy config file and run tests.
		"""
		for i in [0, 1]:
			if i == 0:
				self.create_dummy_outputs0()
			elif i == 1:
				self.create_dummy_outputs1()
			sys_config = read_json( f"{self.dummy_conf_file}{i}.json" )
			sys_config = mlc.ConfigDict( sys_config )
			sys_config = sys_config["System_X"]
			sys_name = sys_config.name

			data_gathering = DataGathering( 
									sys_name = sys_name,
									base_dir = "./",
									fasta_dir = "./",
									sys_config = sys_config
			 )

			self.test_create_full_system( data_gathering )
			print( "\n-----------------------------------\n" )
			self.test_create_residue_index_mapping( data_gathering )
			print( "\n-----------------------------------\n" )
			self.test_create_entity_chain_mapping( data_gathering )
			print( "\n-----------------------------------\n" )
			self.test_create_chain_entity_mapping( data_gathering )
			print( "\n-----------------------------------\n" )
			self.test_account_for_ambiguity( data_gathering )
			print( "\n-----------------------------------\n" )
			self.test_map_residue_to_index( data_gathering )
			print( "\n-----------------------------------\n" )

			print( "\n\n--------------------------------------------------------" )
			print( "--------------------------------------------------------\n\n" )


	def create_dummy_outputs0( self ):
		"""
		Create dummy outputs for input0 for the methods of DataGathering() to be tested.
		See input in sys_conf_dummy0.json.
		"""
		#-------------------------------------------------------
		self.dummy_sys_dict = {
			"XXXX0_1_A":{
				"seq": "ABCDEFGHIJ",
				"positions": [1, 10]
				},
			"XXXX0_1_B": {
				"seq": "ABCDEFGHIJ",
				"positions": [1, 10]
				},
			"XXXX0_2_C": {
				"seq": "KLMNO",
				"positions": [1, 5]
				}
			}

		#-------------------------------------------------------
		self.dummy_res_idx_map = {
			"A": {
				"res_to_ind": {
					1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9
				},
				"ind_to_res": {
					0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10
				}
			},
			"B": {
				"res_to_ind": {
					1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18, 10: 19
				},
				"ind_to_res": {
					10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8, 18: 9, 19: 10
				}
			},
			"C": {
				"res_to_ind": {
					1: 20, 2: 21, 3: 22, 4: 23, 5: 24
				},
				"ind_to_res": {
					20: 1, 21: 2, 22: 3, 23: 4, 24: 5
				}
			}
		}

		#-------------------------------------------------------
		self.dummy_chain_entity_map = {
			"A": 1,
			"B": 1,
			"C": 2
		}

		#-------------------------------------------------------
		self.dummy_entity_chain_map = {
			1: ["A", "B"],
			2: ["C"]
		}

		#-------------------------------------------------------
		# This is the kind of input we expect.
		dummy_xl_data = {
			"prot1": ["prot_1", "prot_1", "prot_1", "prot_1"],
			"res1": [1, 3, 5, 8],
			"prot2": ["prot_1", "prot_2", "prot_1", "prot_2"],
			"res2": [1, 2, 9, 4],
		}
		self.dummy_xl_df = pd.DataFrame( dummy_xl_data )

		# After accounting ambiguity.
		# prot1 -> A, B; prot2 -> C
		# 	For prot1-prot2 we can have: AC, BC, due to ambiguity.
		self.dummy_xl_amb_dict = {
			0: {
				"prot1": ["A"],
				"res1": [1],
				"prot2": ["B"],
				"res2": [1]
			},
			1: {
				"prot1": ["A", "B"],
				"res1": [3, 3],
				"prot2": ["C", "C"],
				"res2": [2, 2]
			},
			2: {
				"prot1": ["A"],
				"res1": [5],
				"prot2": ["B"],
				"res2": [9]
			},
			3: {
				"prot1": ["A", "B"],
				"res1": [8, 8],
				"prot2": ["C", "C"],
				"res2": [4, 4]
			}
		}

		self.dummy_xl_amb_dict_sys = {
			0: {
				"prot1": ["A"],
				"res1": [0],
				"prot2": ["B"],
				"res2": [10]
			},
			1: {
				"prot1": ["A", "B"],
				"res1": [2, 12],
				"prot2": ["C", "C"],
				"res2": [21, 21]
			},
			2: {
				"prot1": ["A"],
				"res1": [4],
				"prot2": ["B"],
				"res2": [18]
			},
			3: {
				"prot1": ["A", "B"],
				"res1": [7, 17],
				"prot2": ["C", "C"],
				"res2": [23, 23]
			}
		}

		# dummy_mapped_xl_amb_dict = {
		# 	"prot1": ["A", "A", "B", "A", "A", "B"],
		# 	"res1":  [0,    2,   12,  4,   7,  17],
		# 	"prot2": ["B", "C", "C", "B", "C", "C"],
		# 	"res2":  [10,   21,  21, 18,   23, 23],
		# }
		# self.dummy_mapped_xl_amb_df = pd.DataFrame( dummy_mapped_xl_amb_data )

		# dummy_xl_amb_data = {
		# 	"prot1": ["A", "B", "A", "B", "A", "B", "A", "B"],
		# 	"res1": [1, 1, 3, 3, 5, 5, 8, 8],
		# 	"prot2": ["B", "A", "C", "C", "B", "A", "C", "C"],
		# 	"res2": [1, 1, 2, 2, 9, 9, 4, 4],
		# }
		# self.dummy_xl_amb_df = pd.DataFrame( dummy_xl_amb_data )

		# dummy_mapped_xl_amb_data = {
		# 	"prot1": ["A", "B", "A", "B", "A", "B", "A", "B"],
		# 	"res1": [0, 10, 2, 12, 4, 14, 7, 17],
		# 	"prot2": ["B", "A", "C", "C", "B", "A", "C", "C"],
		# 	"res2": [10, 0, 21, 21, 18, 8, 23, 23],
		# }
		# self.dummy_mapped_xl_amb_df = pd.DataFrame( dummy_mapped_xl_amb_data )


	def create_dummy_outputs1( self ):
		"""
		Create dummy outputs for input1 for the methods of DataGathering() to be tested.
		See input in sys_conf_dummy1.json.
		"""
		#-------------------------------------------------------
		self.dummy_sys_dict = {
			"XXXX1_1_A":{
				"seq": "HAWAKSAATHSAATH",
				"positions": [1, 15]
				},
			"XXXX1_1_B": {
				"seq": "HAWAKSAATHSAATH",
				"positions": [1, 15]
				},
			"XXXX1_2_C": {
				"seq": "OSAATHICHALL",
				"positions": [1, 12]
				},
			"XXXX1_3_D": {
				"seq": "MUJHELEKESAATHCHALTU",
				"positions": [1, 20]
				},
			"XXXX1_4_E": {
				"seq": "LEHAATHONMEINHAATHCHALTU",
				"positions": [23, 46]
				},
			"XXXX1_5_F": {
				"seq": "OSAATHICHALL",
				"positions": [1, 12]
				},
			"XXXX1_6_G": {
				"seq": "TOHYEMAUSAMHAIBADA",
				"positions": [3, 20]
				},
			"XXXX1_6_H": {
				"seq": "TOHYEMAUSAMHAIBADA",
				"positions": [3, 20]
				},
			"XXXX1_7_I": {
				"seq": "UPARSEAPNADILBHIHAIDEEWANA",
				"positions": [1, 26]
				},
			"XXXX1_8_J": {
				"seq": "TUBANKBADAL",
				"positions": [24, 34]
				}
			}

		#-------------------------------------------------------
		self.dummy_res_idx_map = {
			"A": {
				"res_to_ind": {i: i-1 for i in range( 1, 15+1 )},#15, 0 (length, system_index)
				"ind_to_res": {i-1: i for i in range( 1, 15+1 )}
				},
			"B": {
				"res_to_ind": {i: i+15-1 for i in range( 1, 15+1 )},#15, 15
				"ind_to_res": {i+15-1: i for i in range( 1, 15+1 )}
				},
			"C": {
				"res_to_ind": {i: i+30-1 for i in range( 1, 12+1 )},#13, 30
				"ind_to_res": {i+30-1: i for i in range( 1, 12+1 )}
				},
			"D": {
				"res_to_ind": {i: i+41 for i in range( 1, 20+1 )},#21, 42
				"ind_to_res": {i+41: i for i in range( 1, 20+1 )}
				},
			"E": {
				"res_to_ind": {i: i+61-22 for i in range( 23, 46+1 )},#24, 62
				"ind_to_res": {i+61-22: i for i in range( 23, 46+1 )}
				},
			"F": {
				"res_to_ind": {i: i+85 for i in range( 1, 12+1 )},#13, 86
				"ind_to_res": {i+85: i for i in range( 1, 12+1 )}
				},
			"G": {
				"res_to_ind": {i: i+98-1-2 for i in range( 3, 20+1 )},#21, 98
				"ind_to_res": {i+98-1-2: i for i in range( 3, 20+1 )}
				},
			"H": {
				"res_to_ind": {i: i+116-1-2 for i in range( 3, 20+1 )},#21, 116
				"ind_to_res": {i+116-1-2: i for i in range( 3, 20+1 )}
				},
			"I": {
				"res_to_ind": {i: i+134-1 for i in range( 1, 26+1 )},#26, 134
				"ind_to_res": {i+134-1: i for i in range( 1, 26+1 )}
				},
			"J": {
				"res_to_ind": {i: i+160-1-23 for i in range( 24, 34+1 )},#11, 160
				"ind_to_res": {i+160-1-23: i for i in range( 24, 34+1 )}
				}
		}

		#-------------------------------------------------------
		self.dummy_chain_entity_map = {
			"A": 1, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 6, "I": 7, "J": 8
		}

		#-------------------------------------------------------
		self.dummy_entity_chain_map = {
			1: ["A", "B"], 2: ["C"], 3: ["D"], 4: ["E"], 5: ["F"], 6: ["G", "H"], 7: ["I"], 8: ["J"]
		}

		#-------------------------------------------------------
		# This is the kind of input we expect.
		dummy_xl_data = {
			"prot1": ["prot_1", "prot_1", "prot_1", "prot_1", "prot_2", "prot_2", "prot_3", "prot_3"],
			"res1":  [  1,         3,        5,        8,        2,        6,        4,        9],
			"prot2": ["prot_2", "prot_3", "prot_4", "prot_5", "prot_6", "prot_7", "prot_8", "prot_6"],
			"res2":  [  2,         7,        23,       10,       4,        11,       25,       5],
		}
		self.dummy_xl_df = pd.DataFrame( dummy_xl_data )

		# After accounting ambiguity.
		# prot1 -> A, B; prot2 -> C
		# 	For prot1-prot2 we can have: AC, BC, due to ambiguity.
		self.dummy_xl_amb_dict = {
			0: {
				"prot1": ["A", "B"],
				"res1": [1, 1],
				"prot2": ["C", "C"],
				"res2": [2, 2]
			},
			1: {
				"prot1": ["A", "B"],
				"res1": [3, 3],
				"prot2": ["D", "D"],
				"res2": [7, 7]
			},
			2: {
				"prot1": ["A", "B"],
				"res1": [5, 5],
				"prot2": ["E", "E"],
				"res2": [23, 23]
			},
			3: {
				"prot1": ["A", "B"],
				"res1": [8, 8],
				"prot2": ["F", "F"],
				"res2": [10, 10]
			},
			4: {
				"prot1": ["C", "C"],
				"res1": [2, 2],
				"prot2": ["G", "H"],
				"res2": [4, 4]
			},
			5: {
				"prot1": ["C"],
				"res1": [6],
				"prot2": ["I"],
				"res2": [11]
			},
			6: {
				"prot1": ["D"],
				"res1": [4],
				"prot2": ["J"],
				"res2": [25]
			},
			7: {
				"prot1": ["D", "D"],
				"res1": [9, 9],
				"prot2": ["G", "H"],
				"res2": [5, 5]
			}
		}

		self.dummy_xl_amb_dict_sys = {
			0: {
				"prot1": ["A", "B"],
				"res1": [0, 15],
				"prot2": ["C", "C"],
				"res2": [31, 31]
			},
			1: {
				"prot1": ["A", "B"],
				"res1": [2, 17],
				"prot2": ["D", "D"],
				"res2": [48, 48]
			},
			2: {
				"prot1": ["A", "B"],
				"res1": [4, 19],
				"prot2": ["E", "E"],
				"res2": [62, 62]
			},
			3: {
				"prot1": ["A", "B"],
				"res1": [7, 22],
				"prot2": ["F", "F"],
				"res2": [95, 95]
			},
			4: {
				"prot1": ["C", "C"],
				"res1": [31, 31],
				"prot2": ["G", "H"],
				"res2": [99, 117]
			},
			5: {
				"prot1": ["C"],
				"res1": [35],
				"prot2": ["I"],
				"res2": [144]
			},
			6: {
				"prot1": ["D"],
				"res1": [45],
				"prot2": ["J"],
				"res2": [161]
			},
			7: {
				"prot1": ["D", "D"],
				"res1": [50, 50],
				"prot2": ["G", "H"],
				"res2": [100, 118]
			}
		}


		# dummy_xl_amb_data = {
		# 	"prot1": ["A", "B", "A", "B", "A", "B", "A", "B", "C", "C", "C", "D", "D", "D"],
		# 	"res1":  [1,    1,   3,   3,   5,   5,   8,   8,   2,   2,   6,   4,   9,   9],
		# 	"prot2": ["C", "C", "D", "D", "E", "E", "F", "F", "G", "H", "I", "J", "G", "H"],
		# 	"res2":  [2,    2,   7,   7,   23,  23, 10,  10,   4,   4,   11,  25,  5,   5],
		# }
		# self.dummy_xl_amb_df = pd.DataFrame( dummy_xl_amb_data )

		# dummy_mapped_xl_amb_data = {
		# 	"prot1": ["A", "B", "A", "B", "A", "B", "A", "B", "C", "C", "C", "D", "D", "D"],
		# 	"res1":  [ 0,   15,  2,   17,  4,  19,   7,   22,  31,  31,  35,  45,  50,  50],
		# 	"prot2": ["C", "C", "D", "D", "E", "E", "F", "F", "G", "H", "I", "J", "G", "H"],
		# 	"res2":  [ 31,  31, 48,  48,  62,  62,  95,  95,  99,  117, 144, 161, 100, 118],
		# }
		# self.dummy_mapped_xl_amb_df = pd.DataFrame( dummy_mapped_xl_amb_data )


	def test_create_full_system( self, data_gathering: DataGathering ):
		"""
		Sanity check the method create_full_system().
		"""
		print( "Test 1..." )
		system_dict = data_gathering.create_full_system()

		passed = []
		for sn_1, sn_2 in zip( self.dummy_sys_dict, system_dict ):
			if sn_1 != sn_2:
				raise Exception( f"Test failed. System names do not match. " +
								f"{sn_1} -- {sn_2}..." )
				passed.append( False )
			else:
				passed.append( True )

			answer_seq = self.dummy_sys_dict[sn_1]["seq"]
			sys_seq = system_dict[sn_2]["seq"]

			if answer_seq != sys_seq:
				raise Exception( f"Test failed. Sequences are incorrect for {sn_1}." +
								f"{answer_seq} != {sys_seq}..." )
				passed.append( False )
			else:
				passed.append( True )

			answer_pos = self.dummy_sys_dict[sn_1]["positions"]
			sys_pos = system_dict[sn_2]["positions"]

			if answer_pos != sys_pos:
				raise Exception( f"Test failed. Chain positions are incorrect for {sn_1}." )
				passed.append( False )
			else:
				passed.append( True )

		if all( passed ):
			print( "Test 1 passed - create_full_system()..." )


	def test_create_residue_index_mapping( self, data_gathering: DataGathering ):
		"""
		Sanity check the method create_residue_index_mapping().
		"""
		print( "Test 2..." )

		res_idx_map = data_gathering.create_residue_index_mapping( self.dummy_sys_dict )

		passed = []
		for i1, i2 in zip( self.dummy_res_idx_map, res_idx_map ):
			if i1 != i2:
				raise Exception( f"Test failed. Chain IDs do not match: {i1}  {i2}..." )
				passed.append( False )
			else:
				passed.append( True )
			for j1, j2 in zip( self.dummy_res_idx_map[i1], res_idx_map[i2] ):
				for k1, k2 in zip( self.dummy_res_idx_map[i1][j1], res_idx_map[i2][j2] ):
					if k1 != k2:
						raise Exception( f"Test failed. Keys do not match:" +
										f" {k1}  {k2} for chain {i1}, {i2}..." )
						passed.append( False )
					else:
						passed.append( True )
					v1 = self.dummy_res_idx_map[i1][j1][k1]
					v2 = res_idx_map[i2][j2][k2]
					if v1 != v2:
						raise Exception( f"Test failed. Values do not match:" +
										f" {v1}  {v2} for chain {i1}, {i2}..." )
						passed.append( False )
					else:
						passed.append( True )

		if all( passed ):
			print( "Test 2 passed - create_residue_index_mapping()..." )


	def test_create_entity_chain_mapping( self, data_gathering: DataGathering ):
		"""
		Sanity check the method create_entity_chain_mapping().
		"""
		print( "Test 3..." )
		chain_entity_map = data_gathering.create_entity_chain_mapping( self.dummy_sys_dict )

		passed = []
		for i1, i2 in zip( self.dummy_entity_chain_map, chain_entity_map ):
			if i1 != i2:
				raise Exception( f"Test failed. Entity IDs do not match: {i1}  {i2}..." )
				passed.append( False )
			else:
				passed.append( True )

			chain_id1 = self.dummy_entity_chain_map[i1]
			chain_id2 = chain_entity_map[i2]

			if chain_id1 != chain_id2:
				raise Exception( f"Test failed. Chain IDs do not match: " +
									f"{chain_id1}  {chain_id2} for chain {i1}...")
				passed.append( False )
			else:
				passed.append( True )

		if all( passed ):
			print( "Test 3 passed - create_entity_chain_mapping()..." )


	def test_create_chain_entity_mapping( self, data_gathering: DataGathering ):
		"""
		Sanity check the method create_chain_entity_mapping().
		"""
		print( "Test 4..." )
		chain_entity_map = data_gathering.create_chain_entity_mapping( self.dummy_sys_dict )

		passed = []
		for i1, i2 in zip( self.dummy_chain_entity_map, chain_entity_map ):
			if i1 != i2:
				raise Exception( f"Test failed. Chain IDs do not match: {i1}  {i2}..." )
				passed.append( False )
			else:
				passed.append( True )

			entity_id1 = self.dummy_chain_entity_map[i1]
			entity_id2 = chain_entity_map[i2]

			if entity_id1 != entity_id2:
				raise Exception( f"Test failed. Entity IDs do not match: " +
									f"{entity_id1}  {entity_id2} for chain {i1}...")
				passed.append( False )
			else:
				passed.append( True )

		if all( passed ):
			print( "Test 4 passed - create_chain_entity_mapping()..." )


	def test_account_for_ambiguity( self, data_gathering: DataGathering ):
		"""
		Sanity check the method account_for_ambiguity().
		"""
		print( "Test 5..." )
		data_gathering.res_idx_map = data_gathering.create_residue_index_mapping( self.dummy_sys_dict )
		data_gathering.entity_chain_map = data_gathering.create_entity_chain_mapping( self.dummy_sys_dict )
		data_gathering.chain_entity_map = data_gathering.create_chain_entity_mapping( self.dummy_sys_dict )
		xl_amb_dict = data_gathering.account_for_ambiguity( self.dummy_xl_df )

		passed = self.check_amb_dict( self.dummy_xl_amb_dict, xl_amb_dict )

		if all( passed ):
			print( "Test 5 passed - account_for_ambiguity()..." )



	def test_map_residue_to_index( self, data_gathering: DataGathering ):
		"""
		Sanity check the method map_residue_to_index().
		"""
		print( "Test 6..." )
		data_gathering.res_idx_map = data_gathering.create_residue_index_mapping( self.dummy_sys_dict )
		data_gathering.entity_chain_map = data_gathering.create_entity_chain_mapping( self.dummy_sys_dict )
		data_gathering.chain_entity_map = data_gathering.create_chain_entity_mapping( self.dummy_sys_dict )
		xl_amb_dict = data_gathering.account_for_ambiguity( self.dummy_xl_df )
		xl_amb_dict_sys = data_gathering.map_residue_to_index( xl_amb_dict )

		passed = self.check_amb_dict( self.dummy_xl_amb_dict_sys, xl_amb_dict_sys )

		if all( passed ):
			print( "Test 6 passed - map_residue_to_index()..." )


	def check_amb_dict( self, dummy_amb_dict: Dict, amb_dict: Dict ) -> List:
		"""
		Common function to check both the xl_amb_dict and xl_amb_dict_sys.
		Both have the same form. The former consists of residue positions while
			the latter of system indices.
		"""
		passed = []

		if len( dummy_amb_dict ) != len( amb_dict ):
			raise Exception( "No. of XLs are not the same - " +
							f"{( dummy_amb_dict )} != {len( amb_dict )}...")
			passed.append( False )
		else:
			passed.append( True )

		for dum_pair, pair in zip( dummy_amb_dict, amb_dict ):
			dum_amb_pairs = dummy_amb_dict[dum_pair]
			amb_pairs = amb_dict[pair]

			if len( dum_amb_pairs["prot1"] ) != len( amb_pairs["prot1"] ):
				raise Exception( "Mismatch in the no. of ambiguous pairs created. " +
								f"{len( dum_amb_pairs['prot1'] )}--{len( amb_pairs['prot1'] )}..." )
				passed.append( False )
			else:
				passed.append( True )
			for i in range( len( dum_amb_pairs["prot1"] ) ):
				dum_p1 = dum_amb_pairs["prot1"][i]
				p1 = amb_pairs["prot1"][i]

				if dum_p1 != p1:
					raise Exception( f"Incorrect prot1 - {dum_pair}--{i} - chain ID - {dum_p1} != {p1}" )
					passed.append( False )
				else:
					passed.append( True )

				dum_r1 = dum_amb_pairs["res1"][i]
				r1 = amb_pairs["res1"][i]

				if dum_r1 != r1:
					raise Exception( f"Incorrect prot1 - {dum_pair}--{i} - residue - {dum_r1} != {r1}" )
					passed.append( False )
				else:
					passed.append( True )

				dum_p2 = dum_amb_pairs["prot2"][i]
				p2 = amb_pairs["prot2"][i]

				if dum_p2 != p2:
					raise Exception( f"Incorrect prot2 - {dum_pair}--{i} - chain ID - {dum_p2} != {p2}" )
					passed.append( False )
				else:
					passed.append( True )

				dum_r2 = dum_amb_pairs["res2"][i]
				r2 = amb_pairs["res2"][i]

				if dum_r2 != r2:
					raise Exception( f"Incorrect prot1 - {dum_pair}--{i} - residue - {dum_r2} != {r2}" )
					passed.append( False )
				else:
					passed.append( True )

		return passed



	# def test_account_for_ambiguity( self, data_gathering: DataGathering ):
	# 	"""
	# 	Sanity check the method account_for_ambiguity().
	# 	"""
	# 	print( "Test 5..." )
	# 	data_gathering.res_idx_map = data_gathering.create_residue_index_mapping( self.dummy_sys_dict )
	# 	data_gathering.entity_chain_map = data_gathering.create_entity_chain_mapping( self.dummy_sys_dict )
	# 	data_gathering.chain_entity_map = data_gathering.create_chain_entity_mapping( self.dummy_sys_dict )
	# 	xl_amb_df = data_gathering.account_for_ambiguity( self.dummy_xl_df )

	# 	passed = []
	# 	if self.dummy_xl_amb_df.shape[0] != xl_amb_df.shape[0]:
	# 		raise Exception( "No. of XLs are not the same - " +
	# 						f"{self.dummy_xl_amb_df.shape[0]} != {xl_amb_df.shape[0]}...")
	# 		passed.append( False )
	# 	else:
	# 		passed.append( True )

	# 	for i in range( self.dummy_xl_amb_df.shape[0] ):
	# 		dum_p1 = self.dummy_xl_amb_df.loc[i, "prot1"]
	# 		p1 = xl_amb_df.loc[i, "prot1"]
	# 		if dum_p1 != p1:
	# 			raise Exception( f"Incorrect prot1 chain ID - {dum_p1} != {p1}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_r1 = self.dummy_xl_amb_df.loc[i, "res1"]
	# 		r1 = xl_amb_df.loc[i, "res1"]
	# 		if dum_r1 != r1:
	# 			raise Exception( f"Incorrect residue position for prot1 - {dum_r1} != {r1}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_p2 = self.dummy_xl_amb_df.loc[i, "prot2"]
	# 		p2 = xl_amb_df.loc[i, "prot2"]
	# 		if dum_p2 != p2:
	# 			raise Exception( f"Incorrect prot2 chain ID - {dum_p2} != {p2}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_r2 = self.dummy_xl_amb_df.loc[i, "res2"]
	# 		r2 = xl_amb_df.loc[i, "res2"]
	# 		if dum_r2 != r2:
	# 			raise Exception( f"Incorrect residue position for prot2 - {dum_r2} != {r2}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 	if all( passed ):
	# 		print( "Test 5 passed - account_for_ambiguity()..." )


	# def test_map_residue_to_index( self, data_gathering: DataGathering ):
	# 	"""
	# 	Sanity check the method map_residue_to_index().
	# 	"""
	# 	print( "Test 6..." )
	# 	data_gathering.res_idx_map = data_gathering.create_residue_index_mapping( self.dummy_sys_dict )
	# 	data_gathering.entity_chain_map = data_gathering.create_entity_chain_mapping( self.dummy_sys_dict )
	# 	data_gathering.chain_entity_map = data_gathering.create_chain_entity_mapping( self.dummy_sys_dict )
	# 	xl_amb_df = data_gathering.account_for_ambiguity( self.dummy_xl_df )
	# 	mapped_xl_amb_df = data_gathering.map_residue_to_index( xl_amb_df )

	# 	passed = []
	# 	if self.dummy_mapped_xl_amb_df.shape[0] != mapped_xl_amb_df.shape[0]:
	# 		raise Exception( "No. of XLs are not the same - " +
	# 						f"{self.dummy_mapped_xl_amb_df.shape[0]} != {mapped_xl_amb_df.shape[0]}...")
	# 		passed.append( False )
	# 	else:
	# 		passed.append( True )

	# 	for i in range( self.dummy_mapped_xl_amb_df.shape[0] ):
	# 		dum_p1 = self.dummy_mapped_xl_amb_df.loc[i, "prot1"]
	# 		p1 = mapped_xl_amb_df.loc[i, "prot1"]
	# 		if dum_p1 != p1:
	# 			raise Exception( f"Incorrect prot1 chain ID - {dum_p1} != {p1}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_r1 = self.dummy_mapped_xl_amb_df.loc[i, "res1"]
	# 		r1 = mapped_xl_amb_df.loc[i, "res1"]
	# 		if dum_r1 != r1:
	# 			raise Exception( f"Incorrect system index for prot1 - {dum_r1} != {r1}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_p2 = self.dummy_mapped_xl_amb_df.loc[i, "prot2"]
	# 		p2 = mapped_xl_amb_df.loc[i, "prot2"]
	# 		if dum_p2 != p2:
	# 			raise Exception( f"Incorrect prot2 chain ID - {dum_p2} != {p2}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 		dum_r2 = self.dummy_mapped_xl_amb_df.loc[i, "res2"]
	# 		r2 = mapped_xl_amb_df.loc[i, "res2"]
	# 		if dum_r2 != r2:
	# 			raise Exception( f"Incorrect system index for prot2 - {dum_r2} != {r2}" )
	# 			passed.append( False )
	# 		else:
	# 			passed.append( True )

	# 	if all( passed ):
	# 		print( "Test 6 passed - map_residue_to_index()..." )


TestDataGathering().forward()

