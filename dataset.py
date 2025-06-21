from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from tokenizer import load_tokenizer

tokenizer=load_tokenizer()

class NLIDataset(Dataset):
    def __init__(self, split='train'):
      self.data = load_dataset('sentence-transformers/all-nli', 'pair-class', split=split)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        label=item["label"]
        features=[[item['premise'], item['hypothesis']]]



        self.features=tokenizer.encode_batch(features)
        x = torch.tensor([encoding.ids for encoding in self.features], dtype=torch.long) # Extract ids and convert to tensor
        segment_id = torch.tensor([encoding.type_ids for encoding in self.features], dtype=torch.long) # Extract type_ids and convert to tensor
        mask = torch.tensor([encoding.attention_mask for encoding in self.features], dtype=torch.float) # Extract attention_mask and convert to tensor

        return {"input":x,"segment_id":segment_id,"mask":mask,"label":torch.tensor(label, dtype=torch.long)}


class NQDataset(Dataset):
  def __init__(self,split="train"):
    self.ds = load_dataset("sentence-transformers/natural-questions",split=split)
  def __len__(self):return len(self.ds)
  def __getitem__(self,idx):
    item=self.ds[idx]
    query=item["query"]
    answer=item["answer"]
    data=[query]
    data=tokenizer.encode_batch(data)
    tokens=torch.tensor([encoding.ids for encoding in data],dtype=torch.long)
    segment_id=torch.tensor([encoding.type_ids for encoding in data],dtype=torch.long)
    mask=torch.tensor([encoding.attention_mask for encoding in data],dtype=torch.long)

    passage=tokenizer.encode_batch([answer])
    passage_tokens=torch.tensor([encoding.ids for encoding in passage],dtype=torch.long)
    passage_segment_id=torch.tensor([encoding.type_ids for encoding in passage],dtype=torch.long)
    passage_mask=torch.tensor([encoding.attention_mask for encoding in passage],dtype=torch.long)

    return {"token":tokens,"segment_id":segment_id,"mask":mask,"passage_token":passage_tokens,"passage_segment_id":passage_segment_id,"passage_mask":passage_mask}



