import numpy as np
import torch
import torch.nn as nn
import os


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=24):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count,
                                    embedding_dim=self.dim,
                                    )
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        entities_emb.weight.data[:-1, :].div_(
            entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count,
                                     embedding_dim=self.dim,
                                     )
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(
            relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)
