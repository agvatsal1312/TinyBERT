from dataset import NLIDataset,NQDataset
from torch.utils.data import DataLoader
import config
from model import build_transformer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc

nli_dataset = NLIDataset(split='train')
nli_data_loader = DataLoader(nli_dataset, batch_size=config.nli_batch_size, shuffle=True,pin_memory=True)


nq_dataset = NQDataset(split='train')
nq_data_loader = DataLoader(nq_dataset, batch_size=config.nq_batch_size, shuffle=True,pin_memory=True)


torch.backends.cudnn.benchmark = True 
from torch.amp import  autocast
from torch.amp import GradScaler
scaler=GradScaler()
model,classifier=build_transformer(config.vocab_size,config.n_segments,config.embedd_dim,config.max_len,config.n_layers,config.attn_heads,config.dropout,config.d_ff,config.d_ff,True,2)
model=model.to(config.device)
classifier=classifier.to(config.device)
cls_optimizer=torch.optim.AdamW(classifier.parameters(),lr=config.nli_lr)
optimizer_nli=torch.optim.AdamW(model.parameters(),lr=config.nli_lr)
optimizer_nq=torch.optim.AdamW(model.parameters(),lr=config.nq_lr)

criterion=torch.nn.CrossEntropyLoss()

def mnr_loss(q_emb: torch.Tensor, p_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    # Normalize embeddings
    q = F.normalize(q_emb, p=2, dim=1)
    p = F.normalize(p_emb, p=2, dim=1)

    # Compute cosine similarity matrix: [B, B]
    logits = torch.matmul(q, p.T) / temperature  # scaled dot product

    # Targets are diagonal (i.e., i-th query matches i-th positive)
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Cross entropy loss over in-batch negatives
    return F.cross_entropy(logits, labels)

def train_nli(model,classifier,optimizer_nli,cls_optimizer,criterion,nli_data_loader,nli_epochs):
    best_loss=1e9
    model.train()
    classifier.train()

    for epoch in range(nli_epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(nli_data_loader, desc=f"Epoch {epoch+1}/{config.nli_epochs}")
        for batch in loop:
            x = batch["input"].squeeze(1).to(config.device)
            segment_id = batch["segment_id"].squeeze(1).to(config.device)
            mask = batch["mask"].squeeze(1).to(config.device)
            labels = batch["label"].to(config.device).unsqueeze(1)  # shape: [B, 1]
            
            labels=labels.squeeze(1)
            with autocast(config.device):
                logits = model(x, segment_id, mask)  # shape: [B, 1]
                logits=classifier(logits)
                loss = criterion(logits, labels)

            optimizer_nli.zero_grad()
            cls_optimizer.zero_grad()
            scaler.scale(loss).backward()
    
            scaler.step(optimizer_nli)
            scaler.step(cls_optimizer)
            scaler.update()
            

            train_loss += loss.item()

            # Metrics
            preds = (torch.argmax(logits,dim=-1)).float()
            correct = (preds == labels).float().mean()
            total = labels.size(0)

            loop.set_postfix(Loss=loss.item(), Accuracy=correct)
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = train_loss / len(nli_data_loader)
        accuracy = correct / total
        torch.save({"model":model.state_dict(),"classifier":classifier.state_dict()},f"models/phase_1_last.pt")
        if(avg_loss<best_loss):
            torch.save({"model":model.state_dict(),"classifier":classifier.state_dict()},f"models/phase_1_best.pt")
            best_loss=avg_loss

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print("NLI Training Complete ✅")




def train_nq(model,optimizer_nq,nq_data_loader,nq_epochs):
    accum_steps = 2  # Simulate batch size 64 (32 × 2)
    model.train()
    best_loss = 1e9
    for epoch in range(nq_epochs):
        train_loss = 0.0
        # loop = tqdm(nq_data_loader, desc=f"Epoch {epoch+1}/{nq_epochs}")
        loop = tqdm(enumerate(nq_data_loader), total=len(nq_data_loader), desc=f"Epoch {epoch+1}/10")
        for step,batch in loop:
            query_token = batch["token"].squeeze(1).to(config.device)
            query_segment_id = batch["segment_id"].squeeze(1).to(config.device)
            query_mask = batch["mask"].squeeze(1).to(config.device)

            passage_token=batch["passage_token"].squeeze(1).to(config.device)
            passage_segment_id=batch["passage_segment_id"].squeeze(1).to(config.device)
            passage_mask=batch["passage_mask"].squeeze(1).to(config.device)

            with autocast(config.device):
                q_emb = model(query_token, query_segment_id, query_mask)
                p_emb = model(passage_token, passage_segment_id, passage_mask)

                loss = mnr_loss(q_emb, p_emb) / accum_steps  # Normalize loss
            # q_emb=model(query_token,query_segment_id,query_mask)
            # p_emb=model(passage_token,passage_segment_id,passage_mask)

            # loss = criterion(q_emb,p_emb)
            scaler.scale(loss).backward()
 
            if (step + 1) % accum_steps == 0 or (step + 1) == len(nq_data_loader):
                scaler.step(optimizer_nq)
                scaler.update()
                optimizer_nq.zero_grad()           
                # optimizer_nq.zero_grad()
                # loss.backward()
                # optimizer.step()
            train_loss += loss.item()

            loop.set_postfix(Loss=loss.item())
        gc.collect()
        torch.cuda.empty_cache()
        avg_loss = train_loss / len(nq_data_loader)
        torch.save({"model": model.state_dict()}, f"models/phase_2_last.pt")
        if avg_loss < best_loss:
            torch.save({"model": model.state_dict()}, f"models/phase_2_best.pt")
            best_loss = avg_loss

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    print("NQ Training Complete ✅")




def train(model,classifier,optimizer_nli,optimizer_nq,cls_optimizer,criterion,nli_data_loader,nli_epochs,nq_data_loader,nq_epochs):

    print("Training on NLI Dataset")
    train_nli(model,classifier,optimizer_nli,cls_optimizer,criterion,nli_data_loader,nli_epochs)
    print("Training on NQ Dataset")
    train_nq(model,optimizer_nq,nq_data_loader,nq_epochs)
    print("Training Complete ✅")



if __name__ == "__main__":
    train(model,classifier,optimizer_nli,optimizer_nq,cls_optimizer,criterion,nli_data_loader,config.nli_epochs,nq_data_loader,config.nq_epochs)
    print("All checkpoints saved in models folder✅")


