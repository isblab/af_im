import numpy as np
import io

from Bio.PDB import PDBParser
import modelcif
import modelcif.model
import modelcif.dumper
import modelcif.reference
import modelcif.protocol
import modelcif.alignment
import modelcif.qa_metric

from typing import Any, Sequence, Mapping, Optional

from openfold.np import residue_constants
from openfold.data import feature_pipeline


def get_stringio_objects():
	# For writing predicted structures to modelCIF string.
	fh = io.StringIO()
	return fh



def create_modelcif_object( title: str ):
	"""
	Instantiate a modelCIF object to which all predicted structures 
		will be written as separate models.
	"""
	system = modelcif.System( title = title )
	return system



def prep_protein_to_write( self, outputs: Dict, feature_dict: Dict, 
                            feature_processor:feature_pipeline.FeaturePipeline ):
    """
    Convert the predicted protein structure into a Protein object.
    """
    out = {}
    for k in outputs.keys():
        if isinstance( outputs[k], dict ):
            if k not in out.keys():
                out[k] = {}
            for m in outputs[k].keys():
                out[k][m] = outputs[k][m].squeeze( 0 )
        else:
            out[k] = outputs[k].squeeze( 0 )

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



def add_to_modelcif( system: modelcif.System, prot: Protein, epoch: int ):
    """
    # Taken from openfold.np.protein.py
    - Kartik - modified this function to allow writing multiple models to the same CIF file.
    Instead of returning a ModelCIF string, this function will return the modelCIF object.
    
    Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
    will be treated as the same polymer entity. But note that if chains differ in modelled regions,
    no attempt is made at identifying them as a single polymer entity.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      ModelCIF object.
    """
    #  # - Kartik - Using the epoch as Model index.
    model_index = epoch

    restypes = residue_constants.restypes + ["X"]
    atom_types = residue_constants.atom_types

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    n = aatype.shape[0]
    if chain_index is None:
        chain_index = [0 for i in range(n)]

    # system = modelcif.System( title = f"Epoch {epoch}" )

    # Finding chains and creating entities
    seqs = {}
    seq = []
    last_chain_idx = None
    for i in range(n):
        if last_chain_idx is not None and last_chain_idx != chain_index[i]:
            seqs[last_chain_idx] = seq
            seq = []
        seq.append(restypes[aatype[i]])
        last_chain_idx = chain_index[i]
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
    entities_map = {}
    for key, value in unique_seqs.items():
        model_e = modelcif.Entity(key, description='Model subunit')
        for chain_idx in value:
            entities_map[chain_idx] = model_e

    chain_tags = string.ascii_uppercase
    asym_unit_map = {}
    for chain_idx in set(chain_index):
        # Define the model assembly
        chain_id = chain_tags[chain_idx]
        asym = modelcif.AsymUnit(
        		entities_map[chain_idx], details = f"Model {model_index}, subunit {chain_id}", id = chain_id # - Kartik -
        		# entities_map[chain_idx], details='Model subunit %s' % chain_id, id=chain_id
        		)
        asym_unit_map[chain_idx] = asym
    modeled_assembly = modelcif.Assembly( asym_unit_map.values(), name = f"Modeled assembly {model_index}" ) # - Kartik -
    # modeled_assembly = modelcif.Assembly(asym_unit_map.values(), name='Modeled assembly')

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Global pLDDT, mean of per-residue pLDDTs"

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            # Add all atom sites.
            for i in range(n):
                for atom_name, pos, mask, b_factor in zip(
                        atom_types, atom_positions[i], atom_mask[i], b_factors[i]
                ):
                    if mask < 0.5:
                        continue
                    element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit_map[chain_index[i]], type_symbol=element,
                        seq_id=residue_index[i], atom_id=atom_name,
                        x=pos[0], y=pos[1], z=pos[2],
                        het=False, biso=b_factor, occupancy=1.00)

        def add_scores(self):
            # local scores
            plddt_per_residue = {}
            for i in range(n):
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
    model = _MyModel( assembly = modeled_assembly, name = f"Model {model_index}" ) # - Kartik -
    # model = _MyModel(assembly=modeled_assembly, name='Best scoring model')
    model.add_scores()

    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)

    return system

    # Will do this outside this function to allow writing multiple models to a single file.
    # fh = io.StringIO()
    # modelcif.dumper.write(fh, [system])
    # return fh.getvalue()



