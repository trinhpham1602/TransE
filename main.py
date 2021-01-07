from absl import app
from absl import flags
import data
import TransE as TransE_definition
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data as torch_data
from typing import Tuple
import numpy as np
from time import perf_counter
import glob
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.0001, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=1024, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("emb_dim", default=500,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=24,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=1,
                     help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./data/FB15k-237",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")


def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = FLAGS.dataset_path
    train_file = open(path + "/" + "train.txt", "rb")
    valid_file = open(path + "/" + "valid.txt", "rb")
    test_file = open(path + "/" + "test.txt", "rb")
    with open(path + "/" + "all_triples.txt", "wb") as outfile:
        outfile.write(train_file.read())
        outfile.write(valid_file.read())
        outfile.write(test_file.read())

    train_path = os.path.join(path, "all_triples.txt")

    entity2id, relation2id = data.create_mappings(train_path)

    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    train_set = data.KGDataset(train_path, entity2id, relation2id)
    N_triples = train_set.__len__()
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)

    model = TransE_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=emb_dim,
                                     margin=margin,
                                     device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_epoch_id = 1

    print("Start training TransE with ", epochs, "epochs")
    start = perf_counter()

    for epoch in range(start_epoch_id, epochs + 1):
        model.train()
        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))

            positive_triples = torch.stack(
                (local_heads, local_relations, local_tails), dim=1)
            # Preparing negatives.
            # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
            head_or_tail = torch.randint(
                high=2, size=local_heads.size(), device=device)
            random_entities = torch.randint(
                high=len(entity2id), size=local_heads.size(), device=device)
            broken_heads = torch.where(
                head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(
                head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack(
                (broken_heads, local_relations, broken_tails), dim=1)

            optimizer.zero_grad()

            loss, _, _ = model(positive_triples, negative_triples)
            loss.mean().backward()

            optimizer.step()
        print("Finished the epoch: ", epoch)
    end = perf_counter()
    print("Total time to train TransE: ", end - start)
    entities_emb = model.entities_emb.weight.data.cpu().numpy()
    relations_emb = model.relations_emb.weight.data.cpu().numpy()
    np.savetxt("./output/entities_emb.txt", entities_emb)
    np.savetxt("./output/relations_emb.txt", relations_emb)
    print("Done!")


if __name__ == '__main__':
    app.run(main)
