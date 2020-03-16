import torch.utils.data as data


class Dataset(data.Dataset):

    def __len__(self):
        print("run")
        return 50

    def __getitem__(self, item):
        return item


a = Dataset()
b = data.DataLoader(a, shuffle=True, batch_size=15, drop_last=True)


for x in b:
    print(x)
