import torch
from torch import nn
import math
import config



# Embeddings

class InputEmbedding(nn.Module):
  def __init__(self,vocab_size:int,d_model:int)->None:
    super().__init__()
    self.d_model=d_model
    self.vocab_size=vocab_size
    self.embedd=nn.Embedding(vocab_size,self.d_model)
  def forward(self,x):
  #(batch,seq_len)-->(batch,seq_len,d_model)
    return self.embedd(x)*math.sqrt(self.d_model)

class SegmentEmbedding(nn.Module):
  def __init__(self,n_segments:int,d_model:int)->None:
    super().__init__()
    self.segment_embedd=nn.Embedding(n_segments,d_model)
  def forward(self,x):
    return self.segment_embedd(x)

class PositionalEmbedding(nn.Module):
  def __init__(self,seq_len:int,d_model:int,dropout:float)->None:
    super().__init__()
    self.seq_len=seq_len
    self.d_model=d_model
    self.drop=nn.Dropout(dropout)
    pe=torch.zeros(seq_len,d_model)
    position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
    div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(position*div_term)
    pe[:,1::2]=torch.cos(position*div_term)

    pe=pe.unsqueeze(0)  #adding batch dim
    self.register_buffer("pe",pe)
  def forward(self,x):
    x=x+ self.pe[:,:x.shape[1],:].detach()
    return self.drop(x)



class full_embeddings(nn.Module):
  def __init__(self,src_emb:InputEmbedding,pe_emb:PositionalEmbedding,se_emb:SegmentEmbedding,sep_input_id):
    super().__init__()
    self.src_emb=src_emb
    self.pe_emb=pe_emb
    self.se_emb=se_emb
    self.sep_input_id=sep_input_id

  def forward(self,input_ids,segment_ids):
    
    x=self.pe_emb(self.src_emb(input_ids)+self.se_emb(segment_ids))
    return x


# Normalization block,residual block,feedforward block

class LayerNormalization(nn.Module):
  def __init__(self,d_model, eps:float=1e-6)->None:
    super().__init__()
    self.eps=eps
    self.alpha=nn.Parameter(torch.ones(d_model))
    self.bias=nn.Parameter(torch.zeros(d_model))
  def forward(self,x):
    mean=x.mean(dim=-1,keepdim=True)
    std=x.std(dim=-1,keepdim=True)

    return self.alpha*(x-mean)/(std+self.eps) +self.bias


class FeedForwardNetwork(nn.Module):
  def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
    super().__init__()
    self.linear_1=nn.Linear(d_model,d_ff)
    self.linear_2=nn.Linear(d_ff,d_model)
    self.drop=nn.Dropout(dropout)
    # self.relu=nn.ReLU()
  def forward(self,x):
    x=torch.relu(self.linear_1(x))
    x=self.drop(x)
    return self.linear_2(x)


class ResidualConnection(nn.Module):
  def __init__(self,d_model,drop:float)->None:
    super().__init__()
    self.norm=LayerNormalization(d_model)
    self.drop=nn.Dropout(drop)
  def forward(self,x,sublayer):
    return x+self.drop(sublayer(self.norm(x)))



# Attention Block

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model:int,heads:int,dropout:float):
    super().__init__()
    self.heads=heads
    assert d_model % heads==0, "d_model not divisible by heads"
    self.d_k=d_model//heads
    self.heads=heads
    self.q=nn.Linear(d_model,d_model)
    self.k=nn.Linear(d_model,d_model)
    self.v=nn.Linear(d_model,d_model)
    self.w_o=nn.Linear(d_model,d_model)
    self.dropout=nn.Dropout(dropout)

  def attention(self,q,k,v,mask,dropout):
    d_k=q.shape[-1]
    attention_score=q@k.transpose(-2,-1)/math.sqrt(d_k)
    if mask is not None:
      mask = mask.unsqueeze(1).unsqueeze(2)
      attention_score=attention_score.masked_fill(mask==0,-1e9)
    attention_score=torch.softmax(attention_score,dim=-1)
    if dropout is not None:
      attention_score=dropout(attention_score)
    return attention_score@v, attention_score

  def forward(self,q,k,v,mask):
    q=self.q(q)
    k=self.k(k)
    v=self.v(v)

    q=q.view(q.shape[0],q.shape[1],self.heads,self.d_k).transpose(1,2)
    k=k.view(k.shape[0],k.shape[1],self.heads,self.d_k).transpose(1,2)
    v=v.view(v.shape[0],v.shape[1],self.heads,self.d_k).transpose(1,2)

    x,attention_score=self.attention(q,k,v,mask,self.dropout)
    x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.heads*self.d_k)
    x=self.w_o(x)
    return x



# Encoder

class EncoderBlock(nn.Module):
  def __init__(self,d_model,d_ff,dropout,heads):
    super().__init__()
    self.feedfwd=FeedForwardNetwork(d_model,d_ff,dropout)

    self.residual=nn.ModuleList([ResidualConnection(d_model,dropout),ResidualConnection(d_model,dropout)])
    self.attention=MultiHeadAttention(d_model,heads,dropout)
  def forward(self, x,mask):
    x=self.residual[0](x,lambda x: self.attention(x,x,x,mask))
    x=self.residual[1](x,self.feedfwd)
    return x


class Encoder(nn.Module):
  def __init__(self,d_model,d_ff,dropout,heads,n_layers):
    super().__init__()
    self.encoders=nn.ModuleList([EncoderBlock(d_model,d_ff,dropout,heads) for _ in range(n_layers)])
  def forward(self,x,mask):
    for layer in self.encoders:
      x=layer(x,mask)
    return x



# Classifier

class Classifier(nn.Module):
  def __init__(self,d_model:int,d_ff:int,dropout:float,output_size:int=3)->None:
    super().__init__()
    self.classifier=nn.Sequential(nn.Linear(d_model,d_ff),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_ff,1024),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(1024,output_size),
                                
    )
  def forward(self,x):
    return self.classifier(x)



# Transformer

class Transformer(nn.Module):
  def __init__(self,encoder:Encoder,embeddings:full_embeddings)->None:
    super().__init__()
    self.encoder=encoder
    # self.classifier=classifier
    self.emb=embeddings
  def forward(self, x,segment_ids,mask):
    # B,S,E=x.shape
    x=self.emb(x,segment_ids)
    output=self.encoder(x,mask)
    cls=output[:,0,:]
    cls=cls.squeeze(1)
    # logits=self.classifier(cls)
    return cls
  
  
def build_transformer(vocab_size:int,n_segments:int,embedd_dim:int,max_len:int,n_layers:int,attn_heads:int,dropout:float,d_ff:int,c_d_ff:int,nli_pretrain:bool,sep_input_id:int=2)-> Transformer:
  input_emb=InputEmbedding(vocab_size,embedd_dim)
  seg_emb=SegmentEmbedding(n_segments,embedd_dim)
  pe_emb=PositionalEmbedding(max_len,embedd_dim,dropout)

  full_emb=full_embeddings(input_emb,pe_emb,seg_emb,sep_input_id)

  encoder=Encoder(embedd_dim,d_ff,dropout,attn_heads,n_layers)

  transformer=Transformer(encoder,full_emb)

  for p in transformer.parameters():
    if p.dim()>1:
      nn.init.xavier_uniform_(p)
  if nli_pretrain:
    classifier=Classifier(embedd_dim,d_ff,dropout)
    for p in classifier.parameters():
      if p.dim()>1:
        nn.init.xavier_uniform_(p)
    return transformer, classifier


  return transformer




