import numpy as np
import io
import string
import os
import time

import Bio
from Bio.PDB import PDBParser, Structure, Model, Residue
# import modelcif
# import modelcif.model
# import modelcif.dumper
# import modelcif.reference
# import modelcif.protocol
# import modelcif.alignment
# import modelcif.qa_metric

from typing import Dict, Tuple, Iterator

# from openfold.utils.script_utils import prep_output
# from openfold.np.protein import Protein, get_pdb_headers, _chain_end
# from openfold.np import residue_constants
# from openfold.data import feature_pipeline

# Taken from openfold.np.protein.py
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert(PDB_MAX_CHAINS == 62)



def pdb_to_cif_gemmi( pdb_file_path: str, cif_file_path: str ):
    """
    Convert a .pdb file to a .cif file.

    Input:
    ----------
    pdb_file_path --> Path to the .pdb file.
    cif_file_path --> Path to the .cif file.

    Returns:
    ----------
    None
    """
    struct = gemmi.read_structure( pdb_file_path )

    cif_doc = struct.make_mmcif_document()

    with open( cif_file_path, "w" ) as w:
        w.write( cif_doc.as_string() )



def pdb_to_cif_bio( pdb_file_path: str, cif_file_path: str ):
    """
    Convert a .pdb file to a .cif file.
    Biopython does not preserve all the info while dumping to a .cif file.

    Input:
    ----------
    pdb_file_path --> Path to the .pdb file.
    cif_file_path --> Path to the .cif file.

    Returns:
    ----------
    None
    """
    struct = PDBParser().get_structure( "pdb", pdb_file_path )

    cif = MMCIFIO()
    cif.set_structure( struct )
    cif.save( cif_file_path )



class Parser():
    def __init__( self, pdb_file: str ):
        # I just want a parser class to read from the simulation output file.
        # As we are using PDB format only, CIF compatibility is not required for now.
        self.pdb_file = pdb_file

        # Biopython Structure object.
        self.structure = self.get_structure( 
                                    self.get_parser()
                                     )

    def get_parser( self ) -> Bio.PDB.PDBParser:
        """
        Get the required parser (PDB/CIF) for the input file.
        """
        ext = os.path.splitext( self.pdb_file )[1]

        if "pdb" in ext:
            parser = PDBParser()
        else:
            raise Exception( "Incorrect file format.. Only .pdb format supported for now..." )

        return parser


    def get_structure( self, parser: Bio.PDB.PDBParser ) -> Structure.Structure:
        """
        Return the Biopython Structure object for the input file.
        """
        basename = os.path.basename( self.pdb_file )
        structure = parser.get_structure( basename, self.pdb_file )
        
        return structure



    def get_models( self ) -> Iterator[Model.Model]:
        """
        Yield models in the structure.
        """
        for model in self.structure:
            yield model



    def get_residues( self, model ) -> Iterator[Tuple[Residue.Residue, str]]:
        """
        Get all residues in the model.
        """
        coords = []
        # for model in self.structure:
        for chain in model:
            chain_id = chain.id[0]
            for residue in chain:
                yield residue, chain_id



    def extract_perresidue_quantity( self, residue: Residue, quantity: str ): 
        """
        Given the Biopython residue object, return the specified quantity:
            1. residue position
            2. Ca-coordinate
        """
        symbol = residue.get_resname()
        rep_atom = "CA"

        if quantity == "res_pos":
            return residue.id[1]

        elif quantity == "coords":
            coords = residue[rep_atom].coord
            return coords
        
        else:
            raise Exception( f"Specified quantity: {quantity} does not exist..." )


    def get_coordinates( self ) -> np.array:
        """
        Extract coordinates from all models in the structure.
        """
        for model in self.get_models():
            coords_dict = {}
            
            for residue, chain_id in self.get_residues( model ):
                coords = self.extract_perresidue_quantity( residue, "coords" )
                
                if chain_id not in coords_dict.keys():
                    coords_dict[chain_id] = np.array( coords )
                else:
                    coords_dict[chain_id] = np.append( coords_dict[chain_id], coords )

            coords_dict = {k: v.reshape( -1, 3 ) for k, v in coords_dict.items()}

            yield coords_dict


'''
class SaveModels():
    def __init__( self, title: str, output_format: str, output_path: str ):
        self.title = title
        self.output_format = output_format
        self.output_path = output_path
        self.entities_map = {}
        self.asym_unit_map = {}

        if self.output_format not in ["pdb", "cif"]:
            raise Exception( "Invalid output format specified. Use 'pdb' or 'cif'... " )


    def initialize_system( self ):
        """
        Instantiate a modelcif.System object.
            Top-level class representing a complete modeled system
        """
        if self.output_format == "pdb":
            self.system = []
        else:
            self.system = self.create_system()
            self.model_group = self.create_model_group()
            # Add model_group to system.
            self.system.model_groups.append( self.model_group )


    def create_system( self ):
        """
        Instantiate a modelCIF.System object to which 
            all predicted structures will be added.
        """
        system = modelcif.System( title = self.title )
        return system


    def create_model_group( self ):
        """
        Instantiate an empty modelcif.model_group object.
        All models in the system will be appended to the same model group.
        """
        model_group = modelcif.model.ModelGroup([], name = f"Trajectory" )
        return model_group


    def prep_protein( self, outputs: Dict, feature_dict: Dict, 
                            feature_processor:feature_pipeline.FeaturePipeline ):
        """
        Convert the predicted protein structure into a Protein object.
        Need to remove the batch dim.
        """
        out = {}
        for k in outputs.keys():
            if isinstance( outputs[k], dict ):
                if k not in out.keys():
                    out[k] = {}
                for m in outputs[k].keys():
                    out[k][m] = outputs[k][m].squeeze( 0 ).detach().cpu().numpy()
            else:
                out[k] = outputs[k].squeeze( 0 ).detach().cpu().numpy()

        unrelaxed_protein = prep_output(
            out,                     # out,
            feature_dict,       # batch,
            feature_dict,       # feature_dict,
            feature_processor,  # feature_processor
            config_preset = None,
            multimer_ri_gap = 1,
            subtract_plddt = False
        )

        return unrelaxed_protein


    def create_attributes( self, prot: Protein ):
        """
        Create the required attributes form the Protein object.
        Add entities and asym units to the system for all models.
        """
        self.get_protein_attributes( prot )
        if self.output_format == "cif":
            self.create_entities()
            self.create_asym_units()



    def get_protein_attributes( self, prot: Protein ):
        """
        Obtain the required attributes from the Protein object.
        """
        self.restypes = residue_constants.restypes + ["X"]
        self.atom_types = residue_constants.atom_types

        self.atom_mask = prot.atom_mask
        self.aatype = prot.aatype
        self.atom_positions = prot.atom_positions
        self.residue_index = prot.residue_index.astype(np.int32)
        self.b_factors = prot.b_factors
        self.chain_index = prot.chain_index
        self.n = self.aatype.shape[0]

        if self.chain_index is None:
            self.chain_index = [0 for i in range( self.n )]



    def create_entities( self ):
        """
        Select unique sequences and add them as entities across all models in the system.
        """
        # system = modelcif.System( title = f"Epoch {epoch}" )

        # Finding chains and creating entities
        seqs = {}
        seq = []
        last_chain_idx = None

        for i in range( self.n ):
            if last_chain_idx is not None and last_chain_idx != self.chain_index[i]:
                seqs[last_chain_idx] = seq
                seq = []
            seq.append(self.restypes[self.aatype[i]])
            last_chain_idx = self.chain_index[i]
        # finally add the last chain
        seqs[last_chain_idx] = seq

        # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
        unique_seqs = {}
        for chain_idx, seq_list in seqs.items():
            seq = "".join(seq_list)
            if seq in unique_seqs:
                unique_seqs[seq].append(chain_idx)
            else:
                unique_seqs[seq] = [chain_idx]

        # adding 1 entity per unique sequence
        # entities_map = {}
        for key, value in unique_seqs.items():
            # model_e = modelcif.Entity( key, description = f"Model subunit" )
            if key not in self.entities_map.keys():
                model_e = modelcif.Entity( key, description = f"Model subunit" )
                for chain_idx in value:
                    self.entities_map[chain_idx] = model_e



    def create_asym_units( self ):
        """
        Create a asym units for all entities.
        """

        chain_tags = string.ascii_uppercase
        # asym_unit_map = {}
        for chain_idx in set( self.chain_index ):
            if chain_idx not in self.asym_unit_map.keys():
                # Define the model assembly
                chain_id = chain_tags[chain_idx]
                asym = modelcif.AsymUnit(
                        # self.entities_map[chain_idx], details = f"Model subunit {chain_id}", id = chain_id # - Kartik -
                        self.entities_map[chain_idx], details='Model subunit %s' % chain_id, id=chain_id
                        )
                self.asym_unit_map[chain_idx] = asym
        # modeled_assembly = modelcif.Assembly( self.asym_unit_map.values(), name = f"Modeled assembly {model_index}" ) # - Kartik -
        self.modeled_assembly = modelcif.Assembly(self.asym_unit_map.values(), name='Modeled assembly')


    def add_model( self, prot: Protein, epoch: int ):
        """
        For the 1st model:
            Create all required attributes and add to model.
        For others, just add to model.
        """
        if epoch == 0:
            headers = get_pdb_headers(prot)
            if (len(headers) > 0):
                self.system.extend(headers)
        
        self.create_attributes( prot )

        if self.output_format == "pdb":
            self.add_to_pdb( prot = prot, epoch = epoch  )
        else:
            if epoch == 0:
                self.create_entity_asym_unit( prot = prot )
            self.add_to_modelcif( prot = prot, epoch = epoch  )



    def add_to_pdb( self, prot: Protein, epoch: int ):
        """
        Taken from openfold.np.protein.py
        - Kartik - Modified to write multiple models in a PDB file format.

        Converts a `Protein` instance to a PDB string.

        Args:
          prot: The protein to convert to PDB.

        Returns:
          PDB string.
        """
        # - Kartik - Using the epoch as Model index.
        model_index = epoch
        # restypes = residue_constants.restypes + ["X"]
        res_1to3 = lambda r: residue_constants.restype_1to3.get(self.restypes[r], "UNK")
        # atom_types = residue_constants.atom_types

        # For uniformity, pdblines is replaced to self.system.
        # pdb_lines = []

        # atom_mask = prot.atom_mask
        # aatype = prot.aatype
        # atom_positions = prot.atom_positions
        # residue_index = prot.residue_index.astype(np.int32)
        # b_factors = prot.b_factors
        # chain_index = prot.chain_index.astype(np.int32)

        if np.any( self.aatype > residue_constants.restype_num ):
            raise ValueError("Invalid aatypes.")

        # Construct a mapping from chain integer indices to chain ID strings.
        chain_ids = {}
        for i in np.unique( self.chain_index ): # np.unique gives sorted output.
            if i >= PDB_MAX_CHAINS:
                raise ValueError(
                    f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
                )
            chain_ids[i] = PDB_CHAIN_IDS[i]

        # headers = get_pdb_headers(prot)
        # if (len(headers) > 0):
        #     # pdb_lines.extend(headers)
        #     self.system.extend(headers)

        # pdb_lines.append("MODEL     1")
        self.system.append( f"MODEL     {model_index}" )
        # n = aatype.shape[0]
        atom_index = 1
        last_chain_index = self.chain_index[0]
        prev_chain_index = 0
        chain_tags = string.ascii_uppercase

        # Add all atom sites.
        for i in range( self.aatype.shape[0] ):
            # Close the previous chain if in a multichain PDB.
            if last_chain_index != self.chain_index[i]:
                # pdb_lines.append
                self.system.append(
                    _chain_end(
                        atom_index, 
                        res_1to3( self.aatype[i - 1] ), 
                        chain_ids[self.chain_index[i - 1]], 
                        self.residue_index[i - 1]
                    )
                )
                last_chain_index = self.chain_index[i]
                atom_index += 1 # Atom index increases at the TER symbol.

            res_name_3 = res_1to3( self.aatype[i] )
            for atom_name, pos, mask, b_factor in zip(
                self.atom_types, self.atom_positions[i], self.atom_mask[i], self.b_factors[i]
            ):
                if mask < 0.5:
                    continue

                record_type = "ATOM"
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = atom_name[
                    0
                ]  # Protein supports only C, N, O, S, this works.
                charge = ""

                chain_tag = "A"
                if( self.chain_index is not None ):
                    chain_tag = chain_tags[self.chain_index[i]]

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    #TODO: check this refactor, chose main branch version
                    #f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{self.residue_index[i]:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                # pdb_lines.append(atom_line)
                self.system.append( atom_line )
                atom_index += 1

            should_terminate = (i == self.n - 1)
            if( self.chain_index is not None ):
                if(i != self.n - 1 and self.chain_index[i + 1] != prev_chain_index):
                    should_terminate = True
                    prev_chain_index = self.chain_index[i + 1]

            if(should_terminate):
                # Close the chain.
                chain_end = "TER"
                chain_termination_line = (
                    f"{chain_end:<6}{atom_index:>5}      "
                    f"{res_1to3( self.aatype[i]):>3} "
                    f"{chain_tag:>1}{self.residue_index[i]:>4}"
                )
                # pdb_lines.append(chain_termination_line)
                self.system.append( chain_termination_line )
                # atom_index += 1 # I believe this line is a big in OpenFold implementation. - Kartik -
                # This will add an offset of 1 atom after every chain. - Kartik -

                # I don't need it after every chain. - Kartik -
                # if(i != self.n - 1):
                    # "prev" is a misnomer here. This happens at the beginning of
                    # each new chain.
                    # pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))
                    # self.system.extend( get_pdb_headers( prot, prev_chain_index ) )

        # pdb_lines.append("ENDMDL")
        # pdb_lines.append("END")
        self.system.append("ENDMDL")

        # Pad all lines to 80 characters
        # pdb_lines = [line.ljust(80) for line in pdb_lines]
        # return '\n'.join(pdb_lines) + '\n' # Add terminating newline.



    def add_to_modelcif( self, prot: Protein, epoch: int ):
        """
        Taken from openfold.np.protein.py
        - Kartik - modified this function to allow writing multiple models to the same CIF file.
        Instead of returning a ModelCIF string, this function will add a 
            model to a model group and the latter to the system.
        
        Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
        will be treated as the same polymer entity. But note that if chains differ in modelled regions,
        no attempt is made at identifying them as a single polymer entity.

        Args:
          prot: The protein to convert to PDB.

        Returns:
          ModelCIF object.
        """
        # - Kartik - Using the epoch as Model index.
        model_index = epoch

        class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Predicted lddt"

        class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Global pLDDT, mean of per-residue pLDDTs"


        residue_index = self.residue_index
        chain_index = self.chain_index
        atom_types = self.atom_types
        atom_positions = self.atom_positions
        atom_mask = self.atom_mask
        b_factors = self.b_factors
        n = self.n
        asym_unit_map = self.asym_unit_map
        class _MyModel(modelcif.model.AbInitioModel):
            def get_atoms(self):
                # Add all atom sites.
                for i in range( n ):
                    for atom_name, pos, mask, b_factor in zip(
                            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
                    ):
                        if mask < 0.5:
                            continue
                        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                        yield modelcif.model.Atom(
                            asym_unit = asym_unit_map[chain_index[i]], type_symbol = element,
                            seq_id = residue_index[i], atom_id=atom_name,
                            x=pos[0], y = pos[1], z=pos[2],
                            het = False, biso = b_factor, occupancy = 1.00)

            def add_scores(self):
                # local scores
                plddt_per_residue = {}
                for i in range( n ):
                    for mask, b_factor in zip(atom_mask[i], b_factors[i]):
                        if mask < 0.5:
                            continue
                        # add 1 per residue, not 1 per atom
                        if chain_index[i] not in plddt_per_residue:
                            # first time a chain index is seen: add the key and start the residue dict
                            plddt_per_residue[chain_index[i]] = {residue_index[i]: b_factor}
                        if residue_index[i] not in plddt_per_residue[chain_index[i]]:
                            plddt_per_residue[chain_index[i]][residue_index[i]] = b_factor
                plddts = []
                for chain_idx in plddt_per_residue:
                    for residue_idx in plddt_per_residue[chain_idx]:
                        plddt = plddt_per_residue[chain_idx][residue_idx]
                        plddts.append(plddt)
                        self.qa_metrics.append(
                            _LocalPLDDT(asym_unit_map[chain_idx].residue(residue_idx), plddt))
                # global score
                self.qa_metrics.append((_GlobalPLDDT(np.mean(plddts))))

        # Add the model and modeling protocol to the file and write them out:
        model = _MyModel( assembly = self.modeled_assembly, name = f"Model {model_index}" ) # - Kartik -
        # model = _MyModel(assembly=modeled_assembly, name='Best scoring model')
        model.add_scores()

        self.model_group.append( model )
        # model_group = modelcif.model.ModelGroup([model], name = f"Model {model_index}" )
        # self.system.model_groups.append(model_group)


    # Will do this outside this function to allow writing multiple models to a single file.
    # fh = io.StringIO()
    # modelcif.dumper.write(fh, [system])
    # return fh.getvalue()



    def to_mmcif_string( self ):
        """
        For writing predicted structures to modelCIF string.
        """
        fh = io.StringIO()
        modelcif.dumper.write( fh, [self.system] )
        return fh


    def save( self ):
        """
        Save the models on disk as per the format.
        """
        if self.output_format == "pdb":
            # Pad all lines to 80 characters
            self.system.append("END")
            self.system = [line.ljust(80) for line in self.system]
            self.system = "\n".join( self.system ) + "\n"
            with open( f"{self.output_path}.pdb", 'w' ) as fp:
                fp.write( self.system )
        else:
            fh = self.to_mmcif_string()
            
            with open( f"{self.output_path}.cif", 'w' ) as fp:
                fp.write( fh.getvalue() )
'''


