import argparse
import os
import random
import time
import torch
import numpy as np

from data import Vocab, DataLoader
from generator import Generator
from texttable import Texttable
from utils import read_configuration, init_seed, init_device, build_optimizer, move_to_cuda

torch.cuda.set_device(0)


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def train(config):
    tab_printer(config)

    vocabs = dict()
    vocabs['node'] = Vocab(config["node_vocab"])
    vocabs['token'] = Vocab(config["token_vocab"])
    vocabs['type'] = Vocab(config["type_vocab"])
    vocabs['relation'] = Vocab(config["relation_vocab"])

    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    print("building model...")
    model = Generator(vocabs, config, device)
    model = model.to(device)

    print("loading data...")
    train_data = DataLoader(vocabs, config, config["train_batch_size"], usage="train")
    dev_data = DataLoader(vocabs, config, config["dev_batch_size"], usage="valid")

    optimizer = build_optimizer(model.parameters(), config["learner"], config["lr"], config)

    print("starting training...")
    best_loss = None
    for epoch in range(config["start_epoch"], config["epochs"]):
        model.train()
        train_loss = 0
        train_idx = 0
        for bidx, train_batch in enumerate(train_data):
            optimizer.zero_grad()

            # data -> model -> loss
            train_batch_cuda = move_to_cuda(train_batch, device)
            gen_loss = model(train_batch_cuda)
            train_loss += gen_loss.item()
            train_idx += 1

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print("Epoch %d, Batch %d, Gen loss %.3f" % (epoch, bidx, gen_loss.item()))

        train_loss /= train_idx
        train_ppl = np.exp(train_loss)
        print("\nEpoch %d, Average loss %.3f, Average ppl %.3f\n" % (epoch, train_loss, train_ppl))

        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_idx = 0
            for bidx, dev_batch in enumerate(dev_data):
                dev_batch_cuda = move_to_cuda(dev_batch, device)
                gen_loss = model(dev_batch_cuda)
                dev_loss += gen_loss.item()
                dev_idx += 1

            dev_loss = dev_loss / dev_idx
            dev_ppl = np.exp(dev_loss)
            print("\nEpoch %d, Dev Average loss %.3f, Average ppl %.3f\n" % (epoch, dev_loss, dev_ppl))

        if best_loss is None or dev_loss <= best_loss:
            saved_path = os.path.join(config["saved_dir"], config["dataset"], str(epoch))
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            # save model
            torch.save({"config": config, "model": model.state_dict()}, os.path.join(saved_path, 'model.bin'))

            best_loss = dev_loss


def test(config):
    tab_printer(config)

    vocabs = dict()
    vocabs['node'] = Vocab(config["node_vocab"])
    vocabs['token'] = Vocab(config["token_vocab"])
    vocabs['type'] = Vocab(config["type_vocab"])
    vocabs['relation'] = Vocab(config["relation_vocab"])

    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    print("building model...")
    model = Generator(vocabs, config, device)
    model.load_state_dict(torch.load(config["saved_model"])["model"])
    model = model.to(device)

    test_data = DataLoader(vocabs, config, config["test_batch_size"], usage="test")

    model.eval()
    idx = 1
    attribute = []
    generated_text = []
    reference_text = []
    with torch.no_grad():
        for bidx, test_batch in enumerate(test_data):
            test_batch_cuda = move_to_cuda(test_batch, device)
            generate_corpus = model.work(test_batch_cuda, config["decode_strategy"], config["beam_size"],
                                         config["max_time_step"], device)

            attribute.extend(test_batch["attribute"])
            generated_text.extend(generate_corpus)
            reference_text.extend(test_batch["review"])
            print("Finish {}-th batch example.".format(idx))
            idx += 1

    assert len(generated_text) == len(reference_text)
    saved_file = "{}.res".format(config["dataset"])
    saved_file_path = os.path.join(config["output_dir"], saved_file)
    fout = open(saved_file_path, "w")
    for i in range(len(generated_text)):
        fout.write("Attribute: " + " ".join(attribute[i]) + "\n")
        fout.write("Generated text: " + " ".join(generated_text[i]) + "\n")
        fout.write("Reference text: " + " ".join(reference_text[i]) + "\n")
    fout.close()


if __name__ == "__main__":
    config = read_configuration("config.yaml")

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "test":
        test(config)
    else:
        raise ValueError("Mode Error!")


