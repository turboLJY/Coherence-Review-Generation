import torch
from torch import nn
import torch.nn.functional as F
import math
from module import RelationEncoder, GraphTransformer, Transformer, LearnedPositionalEmbedding, SelfAttentionMask, PointerGenerator
from utils import topk_sampling, greedy_search, Beam_Search_Hypothesis
from data import SOS, EOS


class Generator(nn.Module):
    def __init__(self, vocabs, config, device):
        super(Generator, self).__init__()
        self.vocabs = vocabs

        self.node_embeddings = nn.Embedding(vocabs["node"].size, config["node_dim"], vocabs["node"].padding_idx)
        self.token_embeddings = nn.Embedding(vocabs["token"].size, config["token_dim"], vocabs["token"].padding_idx)
        self.position_embeddings = LearnedPositionalEmbedding(config["token_dim"], device)
        self.type_embeddings = nn.Embedding(vocabs["type"].size, config["type_dim"], vocabs["type"].padding_idx)

        self.relation_encoder = RelationEncoder(vocabs["relation"], config["node_dim"], config["rel_dim"],
                                                config["hidden_size"], config["rnn_layers"],
                                                config["path_encoding_method"], config["dropout_ratio"],
                                                config["bidirectional"])
        self.graph_encoder = GraphTransformer(config["graph_layers"], config["node_dim"], config["ffn_embed_dim"],
                                              config["num_heads"], config["dropout_ratio"], config["weights_dropout"])
        self.sentence_decoder = Transformer(config["inference_layers"], config["token_dim"], config["ffn_embed_dim"],
                                            config["num_heads"], config["dropout_ratio"], config["weights_dropout"],
                                            with_external=True)
        self.pointer_generator = PointerGenerator(config["token_dim"], config["node_dim"])

        self.node_embed_norm = nn.LayerNorm(config["node_dim"])
        self.token_embed_norm = nn.LayerNorm(config["token_dim"])
        self.self_attn_mask = SelfAttentionMask(device)

        self.gate_linear = nn.Linear(config["token_dim"], 1, bias=False)
        self.token_projector = nn.Linear(config["token_dim"], vocabs["token"].size, bias=False)
        self.token_projector.weight = self.token_embeddings.weight
        self.loss = nn.NLLLoss(ignore_index=vocabs["token"].padding_idx, reduction="none")
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_linear.weight, std=0.02)

    def encode_step(self, inp):

        concept_repr = self.node_embeddings(inp["nodes"]) + self.type_embeddings(inp["types"])
        concept_repr = self.node_embed_norm(concept_repr)  # bsz * n_concepts * embed_dim

        # equal -> True, bsz * n_concepts
        concept_padding_mask = torch.eq(inp["nodes"], self.vocabs["node"].padding_idx)

        # rel_num * seq_len, rel_num -> rel_num * hidden_size
        relation = self.relation_encoder(inp["relation_bank"], inp["relation_length"])

        # bsz * n_concepts * n_concepts -> bsz * n_concepts * n_concepts * embed_dim
        relation = relation.index_select(0, inp["relations"].view(-1)).view(*inp["relations"].size(), -1)

        # bsz * n_concepts * embed_dim
        concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_padding_mask)

        return concept_repr, concept_padding_mask

    def work(self, data, decode_strategy, beam_size, max_time_step, device):
        with torch.no_grad():
            generate_corpus = []
            bsz = data["nodes"].size(0)
            concept_repr, concept_padding_mask = self.encode_step(data)
            
            for bid in range(bsz):
                concepts = concept_repr[bid, :, :].unsqueeze(0)
                sub_graphs = data["sub_graphs"][bid]
                generate_tokens = []
                prev_token_ids = []

                if decode_strategy == "beam_search":
                    hypothesis = Beam_Search_Hypothesis(
                        beam_size, self.vocabs["token"].token2idx(SOS),
                        self.vocabs["token"].token2idx(EOS), device,
                        self.vocabs["token"].idx2token
                    )

                for gid in range(len(sub_graphs)):
                    generate_tokens.append(SOS)
                    prev_token_ids.append(self.vocabs["token"].token2idx(SOS))
                    padding_mask = torch.ones(1, concepts.size(1) - len(sub_graphs[gid])).bool().to(device)
                    current_graph_mask = torch.BoolTensor(sub_graphs[gid]).unsqueeze(0).to(device)
                    sub_padding_mask = torch.cat([current_graph_mask, padding_mask], dim=-1)
                    tokens_in = torch.LongTensor([prev_token_ids]).to(device)

                    for gen_idx in range(max_time_step):
                        attn_mask = self.self_attn_mask(tokens_in.size(-1)).bool()
                        token_repr = self.token_embeddings(tokens_in) + self.position_embeddings(tokens_in)
                        token_repr = self.token_embed_norm(token_repr)

                        outs = self.sentence_decoder(token_repr,
                                                     self_attn_mask=attn_mask,
                                                     external_memories=concepts,
                                                     external_padding_mask=sub_padding_mask)

                        gen_logits = self.token_projector(outs[:, -1, :].unsqueeze(1))

                        if decode_strategy == "topk_sampling":
                            token_idx = topk_sampling(gen_logits).item()
                        elif decode_strategy == "greedy_search":
                            token_idx = greedy_search(gen_logits).item()
                        elif decode_strategy == "beam_search":
                            tokens_in, concept_repr, current_graph_mask = \
                                hypothesis.step(gen_idx, gen_logits, encoder_output=concept_repr,
                                                encoder_mask=current_graph_mask, input_type='whole')

                        if decode_strategy in ["topk_sampling", "greedy_search"]:
                            if token_idx == self.vocabs["token"].token2idx(EOS):
                                generate_tokens.append(EOS)
                                prev_token_ids.append(token_idx)
                                break
                            else:
                                generate_tokens.append(self.vocabs["token"].idx2token(token_idx))
                                prev_token_ids.append(token_idx)
                                tokens_in = torch.LongTensor([prev_token_ids]).to(device)
                        elif decode_strategy == "beam_search":
                            if hypothesis.stop():
                                break

                    if decode_strategy == "beam_search":
                        generate_tokens, prev_token_ids = hypothesis.generate()

                    if generate_tokens[-1] != EOS:
                        generate_tokens.append(EOS)
                        prev_token_ids.append(self.vocabs["token"].token2idx(EOS))

                generate_corpus.append(generate_tokens)

        return generate_corpus

    def forward(self, data):
        concept_repr, concept_padding_mask = self.encode_step(data)

        token_repr = self.token_embeddings(data["tokens_in"]) + self.position_embeddings(data["tokens_in"])
        token_repr = self.token_embed_norm(token_repr)  # bsz * n_tokens * embed_dim

        # equal -> True, bsz * n_tokens, don't attend to padding symbols
        token_padding_mask = torch.eq(data["tokens_in"], self.vocabs["token"].padding_idx)

        # n_tokens * n_tokens, don't attend to future symbols
        attn_mask = self.self_attn_mask(data["tokens_in"].size(1)).bool()

        # bsz x n_tokens x n_concepts
        sub_padding_mask = data["sub_masks"].type_as(token_padding_mask)

        outs = self.sentence_decoder(token_repr,
                                     self_padding_mask=token_padding_mask, self_attn_mask=attn_mask,
                                     external_memories=concept_repr, external_padding_mask=sub_padding_mask)

        masks = torch.ne(data["tokens_in"], self.vocabs["token"].padding_idx)

        probs = F.log_softmax(self.token_projector(outs), dim=-1)
        gen_loss = self.loss(probs.view(-1, probs.size(-1)), data["tokens_out"].view(-1))
        gen_loss = gen_loss.reshape_as(data["tokens_out"])
        gen_loss = gen_loss.masked_select(masks).mean()

        return gen_loss
