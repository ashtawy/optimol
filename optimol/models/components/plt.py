import torch
import torch.nn as nn
import torch.nn.functional as F
from optimol.models.components.unimol import UniMolModel, base_architecture, NonLinearHead, DistanceHead, GaussianLayer
# Assume that the following modules are imported from the UniMol codebase:
# - BaseUnicoreModel
# - UniMolModel (the encoder model for molecule or pocket)
# - NonLinearHead  (a simple feed-forward head)
# (You might need to adjust the imports based on your project structure.)

class PLT(nn.Module):
    """
    A modified docking model that predicts properties (e.g. binding affinity)
    by encoding the ligand and pocket separately and then combining their global
    representations.
    """
    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        self.args = args

        # Create separate encoders for ligand and pocket as in the docking model.
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.mol_dictionary = mol_dictionary
        self.pocket_dictionary = pocket_dictionary

        # For property prediction, we remove the cross/coordinate decoders.
        # Instead, we add a simple head that takes the concatenated [CLS] tokens.
        # (Here we assume that each encoder outputs representations of dimension args.mol.encoder_embed_dim.)
        combined_dim = 2 * args.mol.encoder_embed_dim
        # You may use any head design you wish (e.g. multiple layers, dropout, etc.)
        self.property_head = NonLinearHead(in_dim=combined_dim, out_dim=1, activation_fn=args.mol.activation_fn)

    @classmethod
    def build_model(cls, args, task):
        """Class method to build the model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_coord,       # these coordinates are not used in the property head
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_coord,    # not used here either
        pocket_src_edge_type,
        **kwargs
    ):
        # -------- Encode the Molecule (Ligand) --------
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, flag='mol')
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        # mol_outputs[0] is the token-level representation; we assume the first token is a [CLS] token.
        mol_encoder_rep = mol_outputs[0]
        mol_cls = mol_encoder_rep[:, 0, :]  # shape: [batch, hidden_dim]

        # -------- Encode the Pocket --------
        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, flag='pocket')
        pocket_outputs = self.pocket_model.encoder(pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias)
        pocket_encoder_rep = pocket_outputs[0]
        pocket_cls = pocket_encoder_rep[:, 0, :]  # shape: [batch, hidden_dim]

        # -------- Combine and Predict Property --------
        # Concatenate the two [CLS] representations:
        combined_rep = torch.cat([mol_cls, pocket_cls], dim=-1)
        # Pass the combined feature vector through the property head.
        property_prediction = self.property_head(combined_rep)  # e.g. output shape: [batch, 1]

        return property_prediction

    def get_dist_features(self, dist, et, flag):
        """
        Compute graph attention biases from distance and edge type inputs.
        This is essentially the same helper as in the docking model but only for a single encoder.
        """
        n_node = dist.size(-1)
        if flag == 'mol':
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
        elif flag == 'pocket':
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
        else:
            raise ValueError("Unknown flag for get_dist_features: " + flag)
        # Permute and reshape to form the attention bias (as in the original UniMol implementations)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)
        return graph_attn_bias