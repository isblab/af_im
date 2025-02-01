import numpy as np
import math
import matplotlib.pyplot as plt

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



def plot_metrics( metric_dict: Dict, file_name: str ):
	"""
	Plot the per epoch loss.
	"""
	total = len( metric_dict )
	col = 2
	row = math.ceil( total/col )

	fig, ax = plt.subplots( row, col, figsize = ( row*8, row*8 ) )

	c, r = 0, 0
	for i, k in enumerate( metric_dict.keys() ):
		loss = metric_dict[k]
		epochs = np.arange( 0, len( loss ), 1)
		ax[r].plot( epochs, loss, label = k )
		ax[r].set_xlabel( "Epochs", fontsize = 16+row*2 )
		ax[r].set_ylabel( f"{k}", fontsize = 16+row*2 )
		ax[r].xaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		ax[r].yaxis.set_tick_params( labelsize = 14+row, length = 8+row, width = row )
		
		r = r+1 if c == 1 else r
		c = 0 if c == 1 else 1

	plt.savefig( file_name, dpi = 300 )
	plt.close()

