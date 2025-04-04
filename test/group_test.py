from data_tools.dataloader import get_user_groups
from utils.distill_utils import save_user_groups, load_user_groups

train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)

prev_user_groups = load_user_groups(args)
if prev_user_groups is None:
    save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
    print("User groups saved.")
else:
    print("Loaded existing user groups.")
