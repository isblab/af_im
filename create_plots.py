import numpy as np
import math
import matplotlib.pyplot as plt

import torch

from typing import Dict


def plot_loss( loss_dict: Dict, file_name: str ):
	"""
	Plot the per epoch loss.
	"""
	total = len( loss_dict )
	col = 2
	row = math.ceil( total/col )

	fig, ax = plt.subplots( row, col, figsize = ( row*8, row*8 ) )

	c, r = 0, 0
	for i, k in enumerate( loss_dict.keys() ):
		loss = loss_dict[k]
		epochs = np.arange( 0, len( loss ), 1)
		ax[r, c].plot( epochs, loss, label = k )
		ax[r, c].set_xlabel( "Epochs", fontsize = 16+row*2 )
		ax[r, c].set_ylabel( f"{k}", fontsize = 16+row*2 )
		ax[r, c].xaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		ax[r, c].yaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		
		r = r+1 if c == 1 else r
		c = 0 if c == 1 else 1

	plt.savefig( file_name, dpi = 300 )
	plt.close()



def plot_scalar_metrics( metric_dict: Dict, file_name: str ):
	"""
	Plot the per epoch loss.
	"""
	total = len( metric_dict )
	col = 2
	row = math.ceil( total/col )

	fig, ax = plt.subplots( row, 1, figsize = ( row*8, row*8 ) )

	c, r = 0, 0
	for i, k in enumerate( metric_dict.keys() ):
		loss = metric_dict[k]
		epochs = np.arange( 0, len( loss ), 1)
		ax.plot( epochs, loss, label = k )
		ax.set_xlabel( "Epochs", fontsize = 16+row*2 )
		ax.set_ylabel( f"{k}", fontsize = 16+row*2 )
		ax.xaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		ax.yaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		
		r = r+1 if c == 1 else r
		c = 0 if c == 1 else 1

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
	print( aggregate_map[m] )
	
	fig, ax = plt.subplots( 1, 2, figsize = ( 20, 10 ) )
	ax[0].imshow( aggregate_map )
	ax[0].set_title( "XLs formed across all epochs" )
	ax[1].imshow( gt_map )
	ax[0].set_title( "Ground truth XL map" )

	plt.savefig( file_name, dpi = 300 )
	plt.close()


