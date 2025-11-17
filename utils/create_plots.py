"""
This script contains functions to create plots.
"""
import math
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

import torch


def create_plot_from_dict( data_dict: Dict, file_name: str ):
	"""
	Plot the per epoch loss.
	"""
	total = len( data_dict )
	col = 2
	row = math.ceil( total/col )

	_, ax = plt.subplots( row, col, figsize = ( row*8, row*8 ) )

	c, r = 0, 0
	for k in data_dict:
		loss = data_dict[k]
		epochs = np.arange( 0, len( loss ), 1)
		ax[r, c].plot( epochs, loss, label = k )
		ax[r, c].set_xlabel( "Epochs", fontsize = 16+row*2 )
		ax[r, c].set_ylabel( f"{k}", fontsize = 16+row*2 )
		ax[r, c].xaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		ax[r, c].yaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )

		r = r+1 if c == 1 else r
		c = 0 if c == 1 else 1

	plt.tight_layout()
	plt.savefig( file_name, dpi = 300 )
	plt.close()



def plot_per_epoch_xl_satisfcation( metric_dict: Dict, file_name: str ):
	"""
	Plot XL satisfaction per epoch.
	"""
	_, ax = plt.subplots( 1, 1, figsize = ( 8, 8 ) )

	xlr_metric = metric_dict["xlr"]
	epochs = np.arange( 0, len( xlr_metric ), 1)
	ax.plot( epochs, xlr_metric, label = "XL" )
	ax.set_xlabel( "Epochs", fontsize = 16 )
	ax.set_ylabel( "XL satisfaction", fontsize = 16 )
	ax.set_ylim( -0.05, 1.05 )
	ax.xaxis.set_tick_params( labelsize = 14, length = 8, width = 1 )
	ax.yaxis.set_tick_params( labelsize = 14, length = 8, width = 1 )

	plt.tight_layout()
	plt.savefig( file_name, dpi = 300 )
	plt.close()



def plot_xl_map( aggregate_map: torch.Tensor, gt_map: torch.Tensor, file_name ) -> None:
	"""
	Create a plot showing:
		Aggregate contact map across all epochs.
		Ground truth xl contact map.
	"""
	aggregate_map = aggregate_map.squeeze( 0 )
	gt_map = gt_map.squeeze( 0 )

	m = aggregate_map != 0
	print( "XLs satisfied across all epochs..." )
	print( aggregate_map[m] )

	_, ax = plt.subplots( 1, 2, figsize = ( 20, 10 ) )
	ax[0].imshow( aggregate_map )
	ax[0].set_title( "XLs formed across all epochs" )
	ax[1].imshow( gt_map )
	ax[0].set_title( "Ground truth XL map" )

	plt.savefig( file_name, dpi = 300 )
	plt.close()
