import models
from ptflops import get_model_complexity_info
from ahfl_config.args import parser, modify_args
from models.model_config import Config

args = parser.parse_args()
args = modify_args(args)
config = Config()

model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})

macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
