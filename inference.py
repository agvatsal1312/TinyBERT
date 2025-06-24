
from model import build_transformer
import config
import torch
from pathlib import Path
from tokenizer import load_tokenizer
import torch.nn.functional as F

def load_model(model_path):
    model=build_transformer(config.vocab_size,config.n_segments,config.embedd_dim,config.max_len,config.n_layers,config.attn_heads,config.dropout,config.d_ff,config.d_ff,False,2)
    if Path(model_path).exists():
        checkpoint=torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        model=model.to(config.device)
        print("model loaded successfully ✅")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model

def encode_texts(texts,model,tokenizer):
    encodings=tokenizer.encode_batch(texts)

    tokens=torch.tensor([encode.ids for encode in encodings]).long()
    segment_id=torch.tensor([encode.type_ids for encode in encodings]).long()
    mask=torch.tensor([encode.attention_mask for encode in encodings]).long()
    with torch.no_grad():
        outputs = model(
            tokens.to(config.device),segment_id.to(config.device),mask.to(config.device)
        )
        embeddings = outputs # [CLS] token
    return embeddings.cpu()

def semantic_search(input_data):
    all_texts = [input_data["query"]] + input_data["chunks"]
    embeddings = encode_texts(all_texts,model,tokenizer)  # shape: (n+1, dim)

    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]
    print(chunk_embs.shape)

    # Cosine similarity
    similarities = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs)
    print(similarities.shape)

    # Top-k most similar chunks
    topk_indices = torch.topk(similarities, input_data["top_k"]).indices.tolist()
    print(topk_indices)
    results = [input_data["chunks"][i] for i in topk_indices]

    return  results



if __name__=="__main__":
    model=load_model(config.model_file_path)
    tokenizer=load_tokenizer()
    data={"query":"How is Jaigarh Fort? Is it a good place to visit?","chunks":["Jaigarh Fort is an amazing place to visit, but there is vehcle problems usually","Pink Pearl,Fun Kingdom are reaaly bad places.","The Fort is situated in between the hills offers scenic beauty."],"top_k":2}     #set query ,chunks , top_k accordinly.
    results=semantic_search(data)
    for i,chunk in enumerate(results):
        print(f"{i+1} Match 👉👉 {chunk}")











