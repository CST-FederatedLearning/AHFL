from task.ahfl import ahfl
from ahfl_config.args import parser, modify_args

args = parser.parse_args()
args = modify_args(args)

if __name__ == '__main__':
    ahfl(args)
