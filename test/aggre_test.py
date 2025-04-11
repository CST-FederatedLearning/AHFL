import pickle as pkl
import os
from test.validate import validate, local_validate

if args.evalmode:
    load_state_dict(args, model)
    if 'global' in args.evalmode:
        validate(model, test_loader, criterion, args)
    elif 'local' in args.evalmode:
        train_args = eval('argparse.' + open(os.path.join(args.save_path, 'args.txt')).readlines()[0])
        client_groups = pkl.load(open(os.path.join(args.save_path, 'client_groups.pkl'), 'rb')) if os.path.exists(os.path.join(args.save_path, 'client_groups.pkl')) else []
        federator = Federator(model, train_args, client_groups)
        local_validate(federator, test_set, val_user_groups, criterion, args, args.batch_size)
    else:
        raise NotImplementedError
