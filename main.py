from rl.experiment import run
from util import args

if __name__ == '__main__':
    # Defaults to dev_dqn, run once, with no param selection
    run(args.sess_name,
        args.times,
        args.param_selection,
        args.line_search)
