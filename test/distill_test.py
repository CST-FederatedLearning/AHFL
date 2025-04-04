import torch.backends.cudnn as cudnn
from utils.distill_utils import load_checkpoint, load_state_dict
from utils.model_utils import KDLoss

model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
criterion = KDLoss(args).cuda() if args.use_valid else KDLoss(args)

if args.resume:
    checkpoint = load_checkpoint(args, load_best=False)
    if checkpoint is not None:
        args.start_round = checkpoint['round'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded, starting from round", args.start_round)

cudnn.benchmark = True
