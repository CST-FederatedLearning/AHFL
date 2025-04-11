from data_tools.dataloader import get_dataloaders, get_datasets

train_set, val_set, test_set = get_datasets(args)
_, val_loader, test_loader = get_dataloaders(args, args.batch_size, (train_set, val_set, test_set))

print("Train set size:", len(train_set))
print("Validation set size:", len(val_set) if val_set else "None")
print("Test set size:", len(test_set))
