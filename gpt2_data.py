from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is expected to be a list of tuples with 7 elements each

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return only the required elements: sents, length, and gold_binary_actions
        sents = self.data[index][0]  # shape: batchsize x sentence_length
        length = self.data[index][1]
        gold_binary_actions = self.data[index][5]
        return sents, length, gold_binary_actions