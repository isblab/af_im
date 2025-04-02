"""
Khwab ho tum ya, Koi haqeeqat,
Kaun ho tum batlao?
Der se kitni dur khadi ho,
Aur kareeb aa jao.

Playing around with different functional forms for a 
	distance restraint to encode XL data.
Check the loss value and the gradients for XL restraint violations.

This script has two tests.
To calculate the loss I first create some fake data - FakeItTillYouMakeIt().
Basically, fake predicted distance maps and ground truth XL contact maps.

RestraintFunctions() implements some loss functions to be tested.

In Test1,
----------
I am testing different loss functions to see the 
	behaviour of the loss and their respective gradients.
Keeping all parameters fixed (see Test1().set_defaults()) I see the effect of:
	Varying no. of XL violation.
	Varying the violated distance.

In Test2,
----------
I want to try loss functions whose loss saturates for large distances.
I want to use such a loss for accounting ambiguity in intercation data (XLs).
I am trying two losses for now:
	Shifted scaled log loss
	Exponential clamp loss

Usage:
----------
python restraint_funtions.py
"""
import math
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import torch
from torch import nn

seed = 1
torch.manual_seed( seed )


######################################################################################
######################################################################################
class FakeItTillYouMakeIt():
	"""
	A class to generate fake predicted distance maps and 
		ground truth XL contact map.
	"""
	def __init__( self ):
		# Size of the pred and xl_res_mask tensor.
		self.size = [10, 10]
		# No. of Xl residue pairs to make.
		self.num_xl_samples = 10
		# Max bound for Xl satisfaction.
		self.xl_max_bound = 35.0
		# No. of XL residue pairs to violate.
		self.num_violate = 5
		# Mean for the distribution of violated distances.
		self.viol_loc = 36.0


	def __str__( self ):
		"""
		Return all parameters as a "-" seperated string.
		"""
		name = f"{self.size[0]}-{self.num_xl_samples}-"
		name += f"{self.xl_max_bound}-{self.num_violate}-"
		name += f"{self.viol_loc}"
		return name


	def sample_gauss( self, mu: float, sigma: float, num_samples: int
						) -> torch.Tensor:
		"""
		Sample num_samples from a Gaussian with mean = mu and 
									standard deviation = sigma.
		"""
		mu = torch.full( ( num_samples, ), mu )
		sigma = torch.full( ( num_samples, ), sigma )
		
		samples = torch.normal( mu, sigma )
		return samples


	def sample_multinom( self, num_pop: int, num_samples: int ) -> torch.Tensor:
		"""
		Sample num_samples from a tensor distribution with 
			linearly aranged num_pop values.
		"""
		distrib = torch.arange( 0, num_pop, 1 ).float()

		samples = torch.multinomial( distrib, num_samples )
		return samples


	def fake_pred( self ) -> torch.Tensor:
		"""
		Create a fake predicted distance map of arbitary size.
		Sample all pairwise distances from a gaussian with mu = 15 and sigma = 2.
		"""
		m, n = self.size
		mu = 15.0
		sigma = 10.0 # 5.0
		
		pred = self.sample_gauss( mu, sigma, m*n )
		pred = pred.reshape( m, n )
		return pred


	def fake_violations( self, pred: torch.Tensor, xl_res_mask: torch.Tensor
							) -> torch.Tensor:
		"""
		Given the fake predicted distance map and the xl_res_mask,
			sample distances larger than xl_max_bound for selected 
			XL'd residue pairs.
		"""
		sigma = 1.0
		violated_dist = self.sample_gauss( self.viol_loc, sigma, self.num_violate )

		xl_idx = torch.where( xl_res_mask == 1 )
		num_xls = len( xl_idx[0] )

		viol_idx = self.sample_multinom( num_xls, self.num_violate )

		idx1 = xl_idx[0][viol_idx]
		idx2 = xl_idx[1][viol_idx]

		pred[idx1, idx2] = violated_dist
		return pred


	def fake_target( self ) -> torch.Tensor:
		"""
		Create a fake binary XL residue mask.
		"""
		m, n = self.size
		xl_res_mask = torch.zeros( [m*n] )
		xl_idx = self.sample_multinom( m*n, self.num_xl_samples )

		xl_res_mask[xl_idx] = 1
		xl_res_mask = xl_res_mask.reshape( m, n )

		return xl_res_mask


	def get_fake_data( self ) -> torch.Tensor:
		"""
		Create fake predicted distance maps and target xl contact map.
		"""
		fake_data = {}
		# Get some predicted dstance map.
		pred = self.fake_pred()
		# Get the ground truth XL contact map.
		xl_res_mask = self.fake_target()

		# Add fake XL violations on top.
		pred = self.fake_violations( pred = pred,
										xl_res_mask = xl_res_mask )
		return pred, xl_res_mask


######################################################################################
######################################################################################
class RestraintFunctions():
	"""
	Defines candidate functions to be used for the XL restraint.
	"""
	def __init__( self ):
		self.loss_name = "rmse"
		self.base_loss = "mse"

		self.loss_fn = {
			"mse": self.mse,
			"rmse": self.rmse,
			"sigmoid_loss": self.sigmoid_loss,
			"log_loss": self.log_loss,
			"shifted_log_loss": self.shifted_log_loss,
			"shifted_scaled_log_loss": self.shifted_scaled_log_loss,
			"log_cosh_loss": self.log_cosh_loss,
			"exp_clamp_loss": self.exp_clamp_loss
		}


	def compute_loss( self, pred: torch.Tensor, xl_res_mask: torch.Tensor,
					xl_max_bound: float, scale_factor: Optional[float] = None ):
		"""
		"""
		loss_fn = self.loss_fn[self.loss_name]
		pred, violations, N = self.prep_pred( pred, xl_res_mask, xl_max_bound )

		if scale_factor is None:
			loss = loss_fn( violations, xl_max_bound, N )
		else:
			loss = loss_fn( violations, xl_max_bound, N, scale_factor )
		loss.backward()
		grad = pred.grad

		loss = loss.clone().detach()
		grad = grad.clone().detach()
		# Calculate the avg gradient for pred.
		grad = torch.sum( grad )/N

		return loss, grad


	def prep_pred( self, pred: torch.Tensor, xl_res_mask: torch.Tensor, 
					xl_max_bound: float
					) -> Tuple[torch.Tensor, float]:
		"""
		Prepare the predicted tensor for loss calculation.
			Apply the xl_res_mask.
			Return the violated residue pairs.
			Return the total no. of XL residue pairs (N).
		"""
		N = torch.sum( xl_res_mask )

		# N.requires_grad_()
		pred.requires_grad_()
		# xl_res_mask.requires_grad_()

		pred_xl = pred*xl_res_mask

		viol_mask = pred_xl > xl_max_bound

		violations = pred_xl[viol_mask]

		return pred, violations, N


	def get_squared_diff( self, pred: torch.Tensor, xl_max_bound: float ):
		"""
		Compute the squared difference betwen predicted and target distances.
		"""
		squared_diff = ( pred - xl_max_bound )**2
		return squared_diff


	def get_absolute_diff( self, pred: torch.Tensor, xl_max_bound: float ):
		"""
		Compute the absolute difference betwen predicted and target distances.
		"""
		absolute_diff = torch.abs( pred - xl_max_bound )
		return absolute_diff


	def mse( self, pred: torch.Tensor, xl_max_bound: float,
				N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		A simple MSE functon to penalize predicted distances from the target.
		Problems:
			Extra large penalies for large deviataions due to squaring.
			Gradients also increase linearly and may cause exploding gradients.
		"""
		squared_diff = self.get_squared_diff( pred, xl_max_bound )
		mse = torch.sum( squared_diff ) / N
		return mse


	def rmse( self, pred: torch.Tensor, xl_max_bound: float, 
				N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		A simple RMSE functon to penalize predicted distances from the target.
		Penalizes deviations less strictly than MSE.
		"""
		mse = self.mse( pred, xl_max_bound, N )
		rmse = torch.sqrt( mse )
		return rmse


	def sigmoid_loss( self, pred: torch.Tensor, xl_max_bound: float, 
					N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		MSE/RMSE functon to penalize predicted distances from the target.
		Apply a sigmoid to squish the value between 0-1.
		Problems:
			Minimum value for this loss would be 0.5 when rmse = 0.
			Prone to vanishing gradients.
		"""
		if self.base_loss == "squared":
			diff = self.get_squared_diff( pred, xl_max_bound )
		elif self.base_loss == "absolute":
			diff = self.get_absolute_diff( pred, xl_max_bound )
		sig_loss = nn.functional.sigmoid( diff )
		sig_loss = torch.sum( sig_loss )/ N

		return sig_loss


	def log_loss( self, pred: torch.Tensor, xl_max_bound: float, 
					N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		MSE/RMSE functon to penalize predicted distances from the target.
		Apply a log transform to dampen large violations.
		Problems:
				Loss becomes negative for mse/rmse < 1.
		"""
		if self.base_loss == "squared":
			diff = self.get_squared_diff( pred, xl_max_bound )
		elif self.base_loss == "absolute":
			diff = self.get_absolute_diff( pred, xl_max_bound )
		log_loss = torch.log( diff )
		log_loss = torch.sum( log_loss )/ N

		return log_loss


	def shifted_log_loss( self, pred: torch.Tensor, xl_max_bound: float, 
							N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		MSE/RMSE functon to penalize predicted distances from the target.
		Apply a log transform to dampen large violations.
		Adding 1 prevents negative values.
		"""
		if self.base_loss == "squared":
			diff = self.get_squared_diff( pred, xl_max_bound )
		elif self.base_loss == "absolute":
			diff = self.get_absolute_diff( pred, xl_max_bound )
		shifted_log_loss = torch.log( 1.0 + diff )
		shifted_log_loss = torch.sum( shifted_log_loss )/ N

		return shifted_log_loss


	def shifted_scaled_log_loss( self, pred: torch.Tensor, xl_max_bound: float, 
									N: float, scale_factor: Optional[float] = 0.5
									) -> float:
		"""
		MSE/RMSE functon to penalize predicted distances from the target.
		Apply a log transform to dampen large violations.
		Adding 1 prevents negative values.
		Scaling factor reduces the magnitude of loss.
			This controls how fats the loss values saturate for larger violations.
		"""
		alpha = torch.tensor( [scale_factor] )
		if self.base_loss == "squared":
			diff = self.get_squared_diff( pred, xl_max_bound )
		elif self.base_loss == "absolute":
			diff = self.get_absolute_diff( pred, xl_max_bound )
		scaled_loss = alpha*diff
		shifted_scaled_log_loss = torch.log( 1.0 + scaled_loss )
		shifted_scaled_log_loss = torch.sum( shifted_scaled_log_loss )/ N

		return shifted_scaled_log_loss


	def log_cosh_loss( self, pred: torch.Tensor, xl_max_bound: float, 
					N: float, scale_factor: Optional[float] = None ) -> float:
		"""
		MSE/RMSE functon to penalize predicted distances from the target.
		Apply a log transform to dampen large violations.
		"""
		diff = pred - xl_max_bound
		cosh_diff =torch.cosh( diff )
		log_cosh_diff = torch.log( cosh_diff )
		log_cosh_loss = torch.sum( log_cosh_diff )/ N

		return log_cosh_loss


	def exp_clamp_loss( self, pred: torch.Tensor, xl_max_bound: float, 
					N: float, scale_factor: Optional[float] = None ) -> float:
		"""

		MSE/RMSE functon to penalize predicted distances from the target.
		Very strict on small distance violations and flattens for larger violations.
		alpha controls how fast it saturates (smaller value, smoother transition).
		"""
		alpha = torch.tensor( [scale_factor] )
		if self.base_loss == "squared":
			diff = self.get_squared_diff( pred, xl_max_bound )
		elif self.base_loss == "absolute":
			diff = self.get_absolute_diff( pred, xl_max_bound )

		exp_soft_clamp_loss = ( 1 - torch.exp( -1*alpha*diff ) )
		exp_soft_clamp_loss = torch.sum( exp_soft_clamp_loss )/ N
		return exp_soft_clamp_loss


######################################################################################
######################################################################################
class Test2():
	"""
	Assess different loss functions to be used for accountng ambiguity.
	"""
	def __init__( self ):
		print( "\nRunning Test2..." )
		print( "----------------------------------------\n" )
		pass


	def forward( self ):
		"""
		"""
		fake_obj, func_obj = self.initialize()
		for base_loss in ["squared", "absolute"]:
			result_dict = self.test( fake_obj, func_obj, base_loss )
			self.create_plot( result_dict, base_loss )


	def initialize( self ):
		"""
		Instantiate the objects of class:
			FakeItTillYouMakeIt
			RestraintFunctions
		"""
		fake_obj = FakeItTillYouMakeIt()
		func_obj = RestraintFunctions()

		return fake_obj, func_obj


	def set_defaults( self, fake_obj: FakeItTillYouMakeIt ):
		"""
		Set default values for all attributes of FakeItTillYouMakeIt.
		"""
		size = 100
		fake_obj.size = [size, size]
		
		num_xl_samples = int( 0.5*( size )**2 )
		fake_obj.num_xl_samples = num_xl_samples

		num_violate = int( 0.5*num_xl_samples )
		fake_obj.num_violate = num_violate

		fake_obj.viol_loc = 36.0

		return fake_obj


	def test( self, fake_obj: FakeItTillYouMakeIt, func_obj: RestraintFunctions,
					base_loss: str ) -> Dict:
		"""
		Test various loss functions for handling ambiguity.
		Basically, I want a loss that saturates for large values of violated distances.
		I want to test a loss function at different levels of xl distance violation.
		With differnt scale_factors (if required by the loss function).
		"""
		result_dict = {}
		for loss_name in ["shifted_scaled_log_loss", "exp_clamp_loss"]:
			print( f"Using loss function: {loss_name}...")
			func_obj.base_loss = base_loss
			func_obj.loss_name = loss_name

			result_dict.update( {loss_name: {}} )

			for scale_factor in torch.arange( 0.05, 1.05, 0.05 ):
				scale_factor = round( scale_factor.item(), 2 )
				# Set all values to default.
				fake_obj = self.set_defaults( fake_obj )
				# Test for different values of num_violate.
				result_dict[loss_name][scale_factor] = {}
				result_dict[loss_name][scale_factor] = {k:[] for k in ["viol_loc", "loss", "grad"]}
				for viol_loc in torch.arange( 36.0, 102.0, 2.0 ):
					fake_obj.viol_loc = viol_loc.item()

					loss, grad = self.calulate( fake_obj, func_obj, scale_factor )
					
					loss = loss.item()
					grad = grad.item()
					result_dict[loss_name][scale_factor]["viol_loc"].append( viol_loc.item() )
					result_dict[loss_name][scale_factor]["loss"].append( loss )
					result_dict[loss_name][scale_factor]["grad"].append( grad )

		return result_dict


	def calulate( self, fake_obj: FakeItTillYouMakeIt,
					func_obj: RestraintFunctions,
					scale_factor = None ):
		"""
		Compute the loss.
		"""
		pred, xl_res_mask = fake_obj.get_fake_data()
		xl_max_bound = fake_obj.xl_max_bound

		loss, grad = func_obj.compute_loss( pred, xl_res_mask, xl_max_bound, scale_factor )

		return loss, grad


	def create_plot( self, result_dict: Dict, base_loss: str ):
		"""
		"""
		print( result_dict.keys() )
		# for qty in ["num_violate", "viol_loc"]:
		for flag in [0, 1]: # loss/grad indicator
			cols = 2
			rows = math.ceil( len( result_dict )/2 )
			_, ax = plt.subplots( rows, cols, figsize = ( 20, 10 ) )
			r, c = 0, 0
			for loss_name in result_dict:
				for scale in result_dict[loss_name]:
					x = result_dict[loss_name][scale]["viol_loc"]
					if flag == 0:
						y = result_dict[loss_name][scale]["loss"]
						ylabel = "Loss"
					else:
						y = result_dict[loss_name][scale]["grad"]
						ylabel = "Grad"

					ax[r].scatter( x, y, label = scale )
					ax[r].plot( x, y )
					ax[r].set_xlabel( "Viol_loc" )
					ax[r].set_ylabel( ylabel )
					ax[r].set_title( loss_name )
					plt.legend()

				r += 1
				# r = r+1 if c == 1 else r
				# c = 0 if c == 1 else 1
			# Change suffix to mse when using that. Change name to mse/rmse.
			plt.savefig( f"./Plot_Test2_{ylabel}_{base_loss}.png", dpi = 300 )
			plt.close()


######################################################################################
######################################################################################
class Test1():
	"""
	Assess different loss functions and their behaviour for restrait violations.
	"""
	def __init__( self ):
		print( "\nRunning Test1..." )
		print( "----------------------------------------\n" )
		pass


	def forward( self ):
		"""
		"""
		fake_obj, func_obj = self.initialize()
		# squared - calculate the squared deviation.
		# absolute - calculate the absolute deviation.
		for base_loss in ["squared", "absolute"]:
			result_dict = self.test( fake_obj, func_obj, base_loss )
			self.create_plot( result_dict, base_loss )


	def initialize( self ):
		"""
		Instantiate the objects of class:
			FakeItTillYouMakeIt
			RestraintFunctions
		"""
		fake_obj = FakeItTillYouMakeIt()
		func_obj = RestraintFunctions()

		return fake_obj, func_obj


	def set_defaults( self, fake_obj: FakeItTillYouMakeIt ):
		"""
		Set default values for all attributes of FakeItTillYouMakeIt.
		"""
		size = 100
		fake_obj.size = [size, size]
		
		num_xl_samples = int( 0.5*( size )**2 )
		fake_obj.num_xl_samples = num_xl_samples

		num_violate = int( 0.5*num_xl_samples )
		fake_obj.num_violate = num_violate

		fake_obj.viol_loc = 36.0

		return fake_obj


	def test( self, fake_obj: FakeItTillYouMakeIt,
				func_obj: RestraintFunctions,
				base_loss: str ):
		"""
		Calculate the loss and avg gradients for predicted distance map 
			for different loss and inputs.
		Just using multiple variations of a base_loss (mse or rmse).
		"""
		result_dict = {}
		for loss_name in ["mse", "rmse", "sigmoid_loss", "log_loss",
							"shifted_log_loss", "shifted_scaled_log_loss",
							"log_cosh_loss"]:
			print( f"Using loss function: {loss_name}...")
			func_obj.base_loss = base_loss
			func_obj.loss_name = loss_name
			# Set all values to default.
			fake_obj = self.set_defaults( fake_obj )
			# Test for different values of num_violate.
			result_dict[loss_name] = {"num_violate": {}}
			result_dict[loss_name]["num_violate"] = {k:[] for k in ["num_violate", "loss", "grad"]}
			for frac_viol in torch.arange( 0.05, 1.05, 0.05 ):
				num_violate = int( frac_viol*fake_obj.num_xl_samples )
				fake_obj.num_violate = num_violate

				loss, grad = self.calulate( fake_obj, func_obj )
				
				loss = loss.item()
				grad = grad.item()
				result_dict[loss_name]["num_violate"]["num_violate"].append( num_violate )
				result_dict[loss_name]["num_violate"]["loss"].append( loss )
				result_dict[loss_name]["num_violate"]["grad"].append( grad )

			fake_obj = self.set_defaults( fake_obj )
			# Test for different values of num_violate.
			result_dict[loss_name].update( {"viol_loc": {}} )
			result_dict[loss_name]["viol_loc"] = {k:[] for k in ["viol_loc", "loss", "grad"]}
			for viol_loc in torch.arange( 36.0, 102.0, 2.0 ):
				fake_obj.viol_loc = viol_loc.item()

				loss, grad = self.calulate( fake_obj, func_obj )
				
				loss = loss.item()
				grad = grad.item()
				result_dict[loss_name]["viol_loc"]["viol_loc"].append( viol_loc.item() )
				result_dict[loss_name]["viol_loc"]["loss"].append( loss )
				result_dict[loss_name]["viol_loc"]["grad"].append( grad )

		return result_dict


	def calulate( self, fake_obj: FakeItTillYouMakeIt,
					func_obj: RestraintFunctions,
					scale_factor = None ):
		"""
		Compute the loss.
		"""
		pred, xl_res_mask = fake_obj.get_fake_data()
		xl_max_bound = fake_obj.xl_max_bound

		loss, grad = func_obj.compute_loss( pred, xl_res_mask, xl_max_bound )

		return loss, grad


	def create_plot( self, result_dict: Dict, base_loss: str ):
		"""
		"""
		print( result_dict.keys() )
		for qty in ["num_violate", "viol_loc"]:
			for flag in [0, 1]:

				cols = 2
				rows = math.ceil( len( result_dict )/2 )
				
				_, ax = plt.subplots( rows, cols, figsize = ( 20, 20 ) )

				r, c = 0, 0
				for key in result_dict:
					loss_name = key
					x = result_dict[loss_name][qty][qty]
					if flag == 0:
						y = result_dict[loss_name][qty]["loss"]
						ylabel = "Loss"
					else:
						y = result_dict[loss_name][qty]["grad"]
						ylabel = "Grad"
					ax[r, c].scatter( x, y, c = "red" )
					ax[r, c].plot( x, y, c = "blue" )
					ax[r, c].set_xlabel( qty.capitalize() )
					ax[r, c].set_ylabel( ylabel )
					ax[r, c].set_title( loss_name )

					r = r+1 if c == 1 else r
					c = 0 if c == 1 else 1
				# Change suffix to mse when using that. Change name to mse/rmse.
				plt.savefig( f"./Plot_Test1_{qty}_{ylabel}_{base_loss}.png", dpi = 300 )
				plt.close()


if __name__ == "__main__":
	Test1().forward()
	Test2().forward()

