from base.Sever import Sever

federator = Sever(model, args)
best_acc1, best_round = federator.fed_train(
    train_set, val_set, (train_user_groups, val_user_groups, test_user_groups),
    criterion, args, args.batch_size, config.training_params[args.data][args.arch]
)

print(f"Best validation accuracy: {best_acc1:.4f} at round {best_round}")
