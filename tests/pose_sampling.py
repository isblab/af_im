from typing import List, Tuple, Dict, Any
import os, copy, pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import StructureBuilder, MMCIFIO, PDBIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from openfold.config import model_config
from openfold.data import feature_pipeline
from openfold.np import protein
from openfold.utils.geometry.quat_rigid import QuatRigid
# from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array

from utils.pdb_utils import ( SaveModels, prep_protein )
from fit_to_data import parse_nested_dict

"""
Read all atom coords from a PDB/CIF file.
Predict rigid transformations, given the init coords:
    Apply the transformation.
    Compute loss.
    Backprop.
    Save the new coords.
Save all poses as models in a single PDB/CIF file.
Also save individual models as separate PDB/CIF files.
"""


class RigidTransformation( nn.Module ):
    """
    Neural network for predicting rigid transformations.
    """
    def __init__( self, n_coords: int, c_hidden: int ):
        super().__init__()
        self.rigid_transform = nn.Sequential(
            nn.Linear( in_features = n_coords, out_features = c_hidden ),
            nn.ReLU(),
            nn.Linear( in_features = c_hidden, out_features = 16 ),
            nn.ReLU()
            )
        self.quaternion = nn.Linear( in_features = 16, out_features = 4, bias = False )
        self.translation = nn.Linear( in_features = 16, out_features = 3, bias = False )
        # self.quat_rigid = QuatRigid(
        #     c_hidden = c_hidden, full_quat, full_quat = full_quat
        #     )

    def forward( self, coords: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the coordinates for each atom, predict a quaternion
            and translation vector.
        coords -> [B, N, A, 3]
        """
        x = torch.mean( coords, dim = ( 0, 1, 2 ) )
        x = self.rigid_transform( x )
        quat = self.quaternion( x )
        trans = self.translation( x )
        return quat, trans

################################################################################
################################################################################
class PoseSampling():
    """
    Learning rigid transformations.
    """
    def __init__( self ):
        self.sys_name = "4rhz"
        if self.sys_name == "8wtd":
            self.ofold_output_file = f"./8wtd_1_A-8wtd_2_B_model_1_multimer_v3_output_dict.pkl"
            self.feature_dict_file = "./8wtd_1_A-8wtd_2_B_model_1_multimer_v3_feature_dict.pkl"
        elif self.sys_name == "4rhz":
            self.ofold_output_file = f"./4rhz_1_A-4rhz_2_B_model_1_multimer_v3_output_dict.pkl"
            self.feature_dict_file = "./4rhz_1_A-4rhz_2_B_model_1_multimer_v3_feature_dict.pkl"


        self.ensemble_dir = f"./{self.sys_name}_ensemble"
        os.makedirs( self.ensemble_dir, exist_ok = True )
        self.ensemble_file = f"./{self.sys_name}_trajectory"

        self.loss = nn.MSELoss()

        # self.writer = PdbWriter( out_file = self.traj_file )


    def forward( self ):
        """
        """
        out = self.load_ofold_dict()
        rigid_bodies = self.get_rigid_bodies( out = out )
        # self.mmcif_dict = self.load_pdb()
        # feats = self.extract_feats_from_cif()
        # protein = self.select_backbone_atoms( feats = feats )
        sampled_poses = self.predict_pose( system = rigid_bodies )
        self.save_model( out = out, sampled_poses = sampled_poses )
        # self.save_coords( sampled_poses = sampled_poses )

    ################################################################################
    def predict_pose( self, system: List[torch.Tensor] ):
        """
        Given the coordinates for the initial configuration, predict a rigid transformation.
        Apply the transformation to obtain a new configuration.
        """
        chainA, chainB = system
        sampled_poses = []
        sampled_poses.append(
            torch.cat( system, dim = 1 )
            )

        model = RigidTransformation( n_coords = 3, c_hidden = 32 )
        optim = Adam( model.parameters(), lr = 1e-2 )

        for e in range( 50 ):
            print( f"Epoch {e}" )
            optim.zero_grad()
            quat, trans = model( coords = chainB )
            # quat, trans = self.get_quat_and_trans( transform = transform )
            R = self.quat_to_rotmat( quat = quat )
            pred_pose = self.apply_transform( coords = chainB, R = R, trans = trans )
            new_prot = [chainA, pred_pose]
            l = self.compute_loss( new_prot )
            l.backward()
            optim.step()

                # new_prot.append( pred_pose.detach().numpy() )
            new_prot = torch.cat( new_prot, dim = 1 ).detach()
            sampled_poses.append( new_prot )
        return sampled_poses


    def compute_loss( self, new_prot: List[torch.Tensor] ):
        """
        """
        ca = 1
        chainA, chainB = new_prot
        if self.sys_name == "8wtd":
            dist = torch.sqrt(
                torch.sum( ( chainA[:,26-1,ca,:] - chainB[:,91-1,ca,:] )**2, dim = -1 )
                )
        elif self.sys_name == "4rhz":
            dist = torch.sqrt(
                torch.sum( ( chainA[:,57-1,ca,:] - chainB[:,97-1,ca,:] )**2, dim = -1 )
                )

        return self.loss( dist, torch.Tensor( [5.0] ) )


    def get_quat_and_trans( self, transform: torch.Tensor ):
        """
        """
        quat = transform[:4]
        trans = transform[ 4:]
        return quat, trans


    def quat_to_rotmat( self, quat: torch.Tensor ):
        """
        """
        w, x, y, z = quat.unbind( -1 )

        inv_norm = torch.rsqrt( torch.clamp( w**2 + x**2 + y**2 + z**2, min = 1e-8 ) )
        w = w * inv_norm
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm

        xx = 1.0 - 2.0 * ( y ** 2 + z ** 2 )
        xy = 2.0 * ( x * y - w * z )
        xz = 2.0 * ( x * z + w * y )
        yx = 2.0 * ( x * y + w * z )
        yy = 1.0 - 2.0 * ( x ** 2 + z ** 2 )
        yz = 2.0 * ( y * z - w * x )
        zx = 2.0 * ( x * z - w * y )
        zy = 2.0 * ( y * z + w * x )
        zz = 1.0 - 2.0 * ( x ** 2 + y ** 2 )

        R = torch.stack( [
            torch.stack( [xx, xy, xz], dim = -1 ),
            torch.stack( [yx, yy, yz], dim = -1 ),
            torch.stack( [zx, zy, zz], dim = -1 )
            ], dim = -2 )
        return R


    def apply_transform( self,
            coords: torch.Tensor,
            R: torch.Tensor,
            trans: torch.Tensor ) -> torch.Tensor:
        """
        """
        # coords_rot = torch.matmul( coords.unsqueeze( -2 ), R ).squeeze( -2 )
        coords_rot = torch.matmul( coords, R )
        coords_new = coords_rot + trans

        return coords_new

    ################################################################################
    def load_ofold_dict( self ):
        with open( self.ofold_output_file, "rb" ) as f:
            out = pkl.load( f )

        for k in out:
            if isinstance( out[k], Dict ):
                for m in out[k]:
                    out[k][m] = torch.tensor( out[k][m] )
            else:
                out[k] = torch.tensor( out[k] )

        out = parse_nested_dict( out, "add_dim" )

        return out


    def get_rigid_bodies( self, out: Dict[str, Any] ):
        """
        """
        rigid_bodies = []
        final_atom_positions = out["final_atom_positions"]
        asym_ids = torch.unique( out["asym_id"][0] )

        for asym_id in asym_ids:
            idx = torch.where( out["asym_id"][0] == asym_id )[0]
            rb = final_atom_positions[:, idx]
            rigid_bodies.append( rb )
        return rigid_bodies

    ################################################################################
    def save_model( self, out: Dict[str, Any], sampled_poses: List[torch.Tensor] ):
        """
        Save to PDB or CIF file.
        """
        self.load_feature_dict()
        # Craete a SaveModel object.
        save_model_obj = SaveModels( title = self.sys_name, 
                                    output_format = "pdb",
                                    ensemble_dir = self.ensemble_dir )
        save_model_obj.initialize_system()
        for model_id, model in enumerate( sampled_poses ):
            outputs = copy.deepcopy( out )
            outputs["final_atom_positions"] = model
            unrelaxed_protein = self.get_protein_object( outputs = outputs )

            self.add_model(
                save_model_obj = save_model_obj,
                unrelaxed_protein = unrelaxed_protein,
                model_id = model_id )

        save_model_obj.save(
            save_model_obj.system,
            self.ensemble_file )


    def load_feature_dict( self ) -> None:
        """
        Load the feature_dict saved as a .pkl file in the system's director.
        """
        self.ofold_config = model_config( "model_1_multimer_v3" )
        self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )

        with open( self.feature_dict_file, "rb" ) as f:
            self.feature_dict = pkl.load( f )


    def get_protein_object( self, outputs: Dict[str, torch.Tensor]
                            ) -> protein.Protein:
        """
        Given the structure module output dict, return an object of class Protein.
        """
        unrelaxed_protein = prep_protein( 
                                    outputs = outputs, 
                                    feature_dict = self.feature_dict, 
                                    feature_processor = self.feature_processor )
        return unrelaxed_protein

    def add_model( self,
                    save_model_obj: SaveModels,
                    unrelaxed_protein: protein.Protein,
                    model_id: int ):
        """
        For pdb: write the model as a pdb string.
        For cif: add the predicted structure as a model to a modelcif object.
        """
        # save_model_obj.add_to_modelcif( unrelaxed_protein, epoch )
        save_model_obj.add_model( prot = unrelaxed_protein, model_id = model_id )




    # def load_pdb( self ):
    #     """
    #     Load the structure as an MMCIF dict.
    #     """
    #     mmcif_dict = MMCIF2Dict( self.struct_file )
    #     return mmcif_dict


    # def extract_feats_from_cif( self ) -> Dict[str, np.array]:
    #     """
    #     Extract the following fields:
    #         Asym ID
    #         Atom type
    #         Coordinates
    #     """
    #     feats = {
    #     "asym_id": np.array( self.mmcif_dict["_atom_site.auth_asym_id"] ),
    #     "atom_type": np.array( self.mmcif_dict["_atom_site.label_atom_id"] ),
    #     "x": np.array( self.mmcif_dict["_atom_site.Cartn_x"], dtype = float ),
    #     "y": np.array( self.mmcif_dict["_atom_site.Cartn_y"], dtype = float ),
    #     "z": np.array( self.mmcif_dict["_atom_site.Cartn_z"], dtype = float )
    #     }
    #     return feats


    # def get_selection_mask( self )
    #     """
    #     Create a binary mask to select the required 
    #     """


    # def select_backbone_atoms( self, feats: Dict[str, np.array] ) -> torch.Tensor:
    #     """
    #     Select chain A.
    #     Select the backbone atoms (N, CA, C) for chain A.
    #     Keep only the first 4 residues.
    #     """
    #     protein = []
    #     for chain in ["A", "B"]:
    #         asym_id = feats["asym_id"]
    #         chain_mask = np.where( asym_id == chain, True, False )
    #         atom_types = ["CA"]

    #         backbone_mask = np.isin( feats["atom_type"], atom_types )

    #         selection_mask = chain_mask*backbone_mask

    #         x = feats["x"][selection_mask]
    #         y = feats["y"][selection_mask]
    #         z = feats["z"][selection_mask]

    #         A = len( atom_types )
    #         # [A, 3]; A -> no. of backbone atoms
    #         coords = np.column_stack( [x, y, z] )
    #         # [A, 3] -> [N, 3, 3]; N -> no. of residues
    #         coords = coords.reshape( -1, A, 3 )
    #         protein.append( torch.Tensor( coords[:20] ) )

    #     # Select only the first 4 atoms.
    #     return protein

    # ################################################################################
    # def save_coords( self, sampled_poses: List[np.array] ):
    #     """
    #     """
    #     for i in range( len( sampled_poses ) ):
    #         self.writer.write_to_model( model_id = i, coords = sampled_poses[i] )

    #     self.writer.save_struct()

################################################################################
################################################################################
class PdbWriter():
    """
    Toy implementation.
    Write coords to a PDB file.
    """
    def __init__( self, out_file: str ):
        self.out_file = out_file

        self.init_builder()


    def write_to_model( self, model_id: int, coords: np.array ):
        """
        Add the coordinates to the structure.
        """
        chains = ["A", "B"]

        self.add_model( model_id = model_id )
        for i, c in enumerate( chains ):
            self.add_chain( chain = c )
            self.add_coords( coords = coords[i] )


    def init_builder( self ):
        """
        Instantiate the StructureBuilder.
        """
        self.builder = StructureBuilder.StructureBuilder()
        self.builder.init_structure( "Test" )


    def add_model( self, model_id: int ):
        self.builder.init_model( model_id )


    def add_chain( self, chain: str ):
        """
        Add a model with the given model_id to the structure.
        """
        self.builder.init_chain( chain )


    def add_coords( self, coords: np.array ):
        """
        Create a fake Gly residue.
        Add the backbone atoms and coordinates to the structure.
        """
        for i, residue_coords in enumerate( coords, start = 1 ):
            self.builder.init_residue( "GLY", " ", i, " " )
            for atom_idx, ( atom_name, xyz ) in enumerate( zip( ["CA"], residue_coords ) ):
                serial_num = ( i -1 )*1 + atom_idx
                self.builder.init_atom( atom_name, xyz, 1.0, 1.0, " ", atom_name, serial_num, "C" )


    def save_struct( self ):
        """
        Save structure as a PDB file.
        """
        structure = self.builder.get_structure()
        io = PDBIO()
        io.set_structure( structure )
        io.save( self.out_file )


if __name__ == "__main__":
    PoseSampling().forward()
