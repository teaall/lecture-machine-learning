import spacy
import matplotlib.pyplot as plt
import torch

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_vocab(sentences, tokenizer):
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    idx = 4
    for sentence in sentences:
        for token in tokenizer(sentence):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def show_positional_encoding(PositionalEncoding):
    emb_size = 512
    max_len = 500

    # Create the positional encoding object
    pos_encoding = PositionalEncoding(emb_size, max_len)

    # Get the positional encoding matrix (removing the batch dimension for visualization)
    encoding_matrix = pos_encoding.encoding.squeeze(0).T
    plt.figure(figsize=(15, 5))
    plt.imshow(encoding_matrix.numpy(), aspect='auto', cmap='viridis')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding')
    plt.show()


def merge_masks(causal_mask, padding_mask, n_head):
    """

    causal_mask: [seq_len_q, seq_len_k]
    padding_mask: [batch_size, seq_len_k]

    Returns:
        mask: [batch_size, n_head, len_q, len_k] or any shape
        # that can be broadcasted to this shape

    """
    if padding_mask is None:
        if causal_mask is None:
            mask = None
        else:
            # [1, len_q, len_k]
            mask = causal_mask.unsqueeze(0)
    else:  # padding_mask is not None
        batch_size = padding_mask.size(0)
        len_k = padding_mask.size(-1)
        padding_mask = \
            padding_mask.view(batch_size, 1, 1, len_k).expand(-1, n_head, -1, -1).reshape(-1, 1, len_k)
        if causal_mask is None:
            # [batch_size * n_head, 1, len_k]
            mask = padding_mask
        else:
            # [1, len_q, len_k] + [batch_size * n_head, 1, len_k]
            # -> [batch_size * n_head, len_q, len_k]
            mask = causal_mask[None] + padding_mask

        # [batch_size, n_head, 1, len_k] or
        # [batch_size, n_head, len_q, len_k]
        mask = mask.view(batch_size, n_head, -1, len_k)

    return mask


# Create masks and padding functions
def generate_square_subsequent_mask(size, device):
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device),
                      diagonal=1)
    return mask


def create_mask(src, tgt, vocab_de, vocab_en, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(
        tgt_seq_len, device)  # [len_tgt, len_tgt]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(
        torch.bool)  # [len_src, len_src]

    src_padding_mask = (src == vocab_de['<pad>'])  # [batch, len_src]
    tgt_padding_mask = (tgt == vocab_en['<pad>'])  # [batch, len_tgt]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
