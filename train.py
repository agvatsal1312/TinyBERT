from dataset import NLIDataset,NQDataset
from torch.utils.data import DataLoader
import config
from model import build_transformer
import torch
import torch.nn.functional as F
from tqdm import tqdm

nli_dataset = NLIDataset(split='train')
nli_data_loader = DataLoader(nli_dataset, batch_size=config.nli_batch_size, shuffle=True)


nq_dataset = NQDataset(split='train')
nq_data_loader = DataLoader(nq_dataset, batch_size=config.nq_batch_size, shuffle=True)



model,classifier=build_transformer(config.vocab_size,config.n_segments,config.embedd_dim,config.max_len,config.n_layers,config.attn_heads,config.dropout,config.d_ff,config.d_ff,True,2)
model=model.to(config.device)
classifier=classifier.to(config.device)
optimizer_nli=torch.optim.Adam(model.parameters(),lr=config.nli_lr)
optimizer_nq=torch.optim.Adam(model.parameters(),lr=config.nq_lr)

criterion=torch.nn.CrossEntropyLoss()

def mnr_loss(q_emb: torch.Tensor, p_emb: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    
    q_norm = F.normalize(q_emb, p=2, dim=1)
    p_norm = F.normalize(p_emb, p=2, dim=1)

    # 2. Compute full similarity matrix: [N, N]
    sim_matrix = q_norm @ p_norm.t()  # cosine similarities

    # 3. For each i:
    pos_scores = sim_matrix.diag().unsqueeze(1)  # [N, 1]
    neg_scores = sim_matrix  # [N, N] (including diagonals)

    # 4. Compute margin-based pairwise loss
    diff = pos_scores - neg_scores + margin
    # Zero out diagonal terms (don't compare with itself)
    diff.fill_diagonal_(0.0)
    loss_per_element = F.relu(diff)  # max(0, ...)
    # 5. Sum over negatives, average over batch
    loss = loss_per_element.sum(dim=1).mean()
    return loss

def train_nli(model,classifier,optimizer,criterion,nli_data_loader,nli_epochs):
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
            

            logits = model(x, segment_id, mask)  # shape: [B, 1]
            logits=classifier(logits)
    
            labels=labels.squeeze(1)
        
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Metrics
            preds = (torch.argmax(logits,dim=-1)).float()
            correct = (preds == labels).float().mean()
            total = labels.size(0)

            loop.set_postfix(Loss=loss.item(), Accuracy=correct)

        avg_loss = train_loss / len(nli_data_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print("NLI Training Complete ✅")




def train_nq(model,optimizer,criterion,nq_data_loader,nq_epochs):
    for epoch in range(nq_epochs):
        train_loss = 0.0
        loop = tqdm(nq_data_loader, desc=f"Epoch {epoch+1}/{nq_epochs}")
        for batch in loop:
            query_token = batch["token"].squeeze(1).to(config.device)
            query_segment_id = batch["segment_id"].squeeze(1).to(config.device)
            query_mask = batch["mask"].squeeze(1).to(config.device)

            passage_token=batch["passage_token"].squeeze(1).to(config.device)
            passage_segment_id=batch["passage_segment_id"].squeeze(1).to(config.device)
            passage_mask=batch["passage_mask"].squeeze(1).to(config.device)


            q_emb=model(query_token,query_segment_id,query_mask)
            p_emb=model(passage_token,passage_segment_id,passage_mask)

            loss = criterion(q_emb,p_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            loop.set_postfix(Loss=loss.item())
        print("NQ Training Complete ✅")

    avg_loss = train_loss / len(nq_data_loader)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")




def train(model,classifier,optimizer,criterion,nli_data_loader,nli_epochs,nq_data_loader,nq_epochs):

    print("Training on NLI Dataset")
    train_nli(model,classifier,optimizer,criterion,nli_data_loader,nli_epochs)
    print("Training on NQ Dataset")
    train_nq(model,optimizer,criterion,nq_data_loader,nq_epochs)
    print("Training Complete ✅")

def save_model(model,classifier):
    torch.save({
        "model":model.state_dict(),
    }, "model.pt")

    torch.save({
        "classifier":classifier.state_dict(),
    }, "classifier.pt")

    print("Model and Classifier Saved ✅")

if __name__ == "__main__":
    train(model,classifier,optimizer_nli,criterion,nli_data_loader,config.nli_epochs,nq_data_loader,config.nq_epochs)
    save_model(model,classifier)


