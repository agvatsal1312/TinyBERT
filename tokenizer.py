import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
import config
import os



from tokenizers.pre_tokenizers import Whitespace

def load_tokenizer():

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    if  os.path.exists(config.tokenizer_file_path):
        tokenizer = Tokenizer.from_file(config.tokenizer_file_path)
    else:
        raise FileNotFoundError("Tokenizer file not found")

    from tokenizers.processors import TemplateProcessing
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),

        ],
    )

    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]",length=config.max_len)
    tokenizer.enable_truncation(max_length=config.max_len)
    return tokenizer