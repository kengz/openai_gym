import argparse
import collections
import inspect
import json
import logging
import multiprocessing as mp
import numpy as np
import re
import sys
from datetime import datetime, timedelta
from os import path, listdir, environ, getpid
from textwrap import wrap

PARALLEL_PROCESS_NUM = mp.cpu_count()
TIMESTAMP_REGEX = r'(\d{4}_\d{2}_\d{2}_\d{6})'
ASSET_PATH = path.join(path.dirname(__file__), 'asset')


# import and safeguard the PROBLEMS, EXPERIMENT_SPECS with checks
def import_guard_asset():
    PROBLEMS = json.loads(open(path.join(ASSET_PATH, 'problems.json')).read())
    EXPERIMENT_SPECS = {}
    spec_files = [spec_json for spec_json in listdir(
        ASSET_PATH) if spec_json.endswith('experiment_specs.json')]
    for filename in spec_files:
        specs = json.loads(open(path.join(ASSET_PATH, filename)).read())
        EXPERIMENT_SPECS.update(specs)

    REQUIRED_PROBLEM_KEYS = [
        'GYM_ENV_NAME', 'SOLVED_MEAN_REWARD',
        'MAX_EPISODES', 'REWARD_MEAN_LEN']
    REQUIRED_SPEC_KEYS = [
        'problem', 'Agent', 'HyperOptimizer',
        'Memory', 'Optimizer', 'Policy', 'PreProcessor', 'param']

    for problem_name, problem in PROBLEMS.items():
        assert all(k in problem for k in REQUIRED_PROBLEM_KEYS), \
            '{} needs all REQUIRED_PROBLEM_KEYS'.format(
            problem_name)

    for experiment_name, spec in EXPERIMENT_SPECS.items():
        assert all(k in spec for k in REQUIRED_SPEC_KEYS), \
            '{} needs all REQUIRED_SPEC_KEYS'.format(experiment_name)
        EXPERIMENT_SPECS[experiment_name]['experiment_name'] = experiment_name
        if 'param_range' not in EXPERIMENT_SPECS[experiment_name]:
            continue

        param_range = EXPERIMENT_SPECS[experiment_name]['param_range']
        for param_key, param_val in param_range.items():
            if isinstance(param_val, list):
                param_range[param_key] = sorted(param_val)
            elif isinstance(param_val, dict):
                pass
            else:
                assert False, \
                    'param_range value must be list or dict: {}.{}:{}'.format(
                        experiment_name, param_key, param_val)

        EXPERIMENT_SPECS[experiment_name]['param_range'] = param_range
    return PROBLEMS, EXPERIMENT_SPECS

PROBLEMS, EXPERIMENT_SPECS = import_guard_asset()


# parse_args to add flag
parser = argparse.ArgumentParser(description='Set flags for functions')
parser.add_argument("-d", "--debug",
                    help="activate debug log",
                    action="store_const",
                    dest="loglevel",
                    const=logging.DEBUG,
                    default=logging.INFO)
parser.add_argument("-q", "--quiet",
                    help="change log to warning level",
                    action="store_const",
                    dest="loglevel",
                    const=logging.WARNING,
                    default=logging.INFO)
parser.add_argument("-b", "--blind",
                    help="dont render graphics",
                    action="store_const",
                    dest="render",
                    const=False,
                    default=True)
parser.add_argument("-e", "--experiment",
                    help="specify experiment to run",
                    action="store",
                    type=str,
                    nargs='?',
                    dest="experiment",
                    default="dev_dqn")
parser.add_argument("-t", "--times",
                    help="number of times session is run",
                    action="store",
                    nargs='?',
                    type=int,
                    dest="times",
                    default=1)
parser.add_argument("-p", "--param_selection",
                    help="run parameter selection if present",
                    action="store_true",
                    dest="param_selection",
                    default=False)
parser.add_argument("-a", "--analyze",
                    help="only run analyze_data",
                    action="store_true",
                    dest="analyze_only",
                    default=False)
parser.add_argument("-x", "--max_episodes",
                    help="manually set environment max episodes",
                    action="store",
                    nargs='?',
                    type=int,
                    dest="max_epis",
                    default=-1)
args = parser.parse_args([]) if environ.get('CI') else parser.parse_args()

# Goddam python logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.setLevel(args.loglevel)
logger.addHandler(handler)
logger.propagate = False
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # mute tf warnings on optimized setup


def log_self(subject):
    max_info_len = 300
    info = '{}, param: {}'.format(
        subject.__class__.__name__,
        to_json(subject.__dict__))
    trunc_info = (
        info[:max_info_len] + '...' if len(info) > max_info_len else info)
    logger.debug(trunc_info)


def wrap_text(text):
    return '\n'.join(wrap(text, 60))


def make_line(line='-'):
    if environ.get('CI'):
        return
    columns = 80
    line_str = line*int(columns)
    return line_str


def log_delimiter(msg, line='-'):
    delim_msg = '''\n{0}\n{1}\n{0}\n\n'''.format(
        make_line(line), msg)
    logger.info(delim_msg)


def log_trial_delimiter(trial, action):
    log_delimiter('{} Trial #{}/{} on PID {}:\n{}'.format(
        action, trial.trial_num, trial.num_of_trials,
        getpid(), trial.trial_id), '=')


def log_session_delimiter(sess, action):
    log_delimiter(
        '{} Session #{}/{} of Trial #{}/{} on PID {}:\n{}'.format(
            action, sess.session_num, sess.num_of_sessions,
            sess.trial.trial_num, sess.trial.num_of_trials,
            getpid(), sess.session_id))


def timestamp():
    '''timestamp used for filename'''
    timestamp_str = '{:%Y_%m_%d_%H%M%S}'.format(datetime.now())
    assert re.search(TIMESTAMP_REGEX, timestamp_str)
    return timestamp_str


def timestamp_elapse(s1, s2):
    '''calculate the time elapsed between timestamps from s1 to s2'''
    FMT = '%Y_%m_%d_%H%M%S'
    delta_t = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return str(delta_t)


def timestamp_elapse_to_seconds(s1):
    a = datetime.strptime(s1, '%H:%M:%S')
    secs = timedelta(hours=a.hour, minutes=a.minute, seconds=a.second).seconds
    return secs


# own custom sorted json serializer, cuz python
def to_json(o, level=0):
    INDENT = 2
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k in sorted(o.keys()):
            v = o[k]
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list) or isinstance(o, tuple):
        ret += "[" + ",".join([to_json(e, level+1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + \
            ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    elif hasattr(o, '__class__'):
        ret += '"' + o.__class__.__name__ + '"'
    else:
        raise TypeError(
            "Unknown type '%s' for json serialization" % str(type(o)))
    return ret


# format object and its properties into printable dict
def format_obj_dict(obj, keys):
    if isinstance(obj, dict):
        return to_json(
            {k: obj.get(k) for k in keys if obj.get(k) is not None})
    else:
        return to_json(
            {k: getattr(obj, k, None) for k in keys
             if getattr(obj, k, None) is not None})


# cast dict to have flat values (int, float, str)
def flat_cast_dict(d):
    for k in d:
        v = d[k]
        if not isinstance(v, (int, float)):
            d[k] = str(v)
    return d


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_module(GREF, dot_path):
    # get module from globals() by string dot_path
    path_arr = dot_path.split('.')
    # base level from globals
    mod = GREF.get(path_arr.pop(0))
    for deeper_path in path_arr:
        mod = getattr(mod, deeper_path)
    return mod


def import_package_files(globals_, locals_, __file__):
    '''
    Dynamically import all the public attributes of the python modules in this
    file's directory (the package directory) and return a list of their names.
    '''
    exports = []
    # globals_, locals_ = globals(), locals()
    package_path = path.dirname(__file__)
    package_name = path.basename(package_path)

    for filename in listdir(package_path):
        modulename, ext = path.splitext(filename)
        if modulename[0] != '_' and ext in ('.py', '.pyw'):
            subpackage = '{}.{}'.format(
                package_name, modulename)  # pkg relative
            module = __import__(subpackage, globals_, locals_, [modulename])
            modict = module.__dict__
            names = (modict['__all__'] if '__all__' in modict else
                     [name for name in
                      modict if inspect.isclass(modict[name])])  # all public
            exports.extend(names)
            globals_.update((name, modict[name]) for name in names)

    return exports


def clean_id_str(id_str):
    return id_str.split('/').pop().split('.').pop(0)


def parse_trial_id(id_str):
    c_id_str = clean_id_str(id_str)
    if re.search(TIMESTAMP_REGEX, c_id_str):
        name_time_trial = re.split(TIMESTAMP_REGEX, c_id_str)
        if len(name_time_trial) == 3:
            return c_id_str
        else:
            return None
    else:
        return None


def parse_experiment_id(id_str):
    c_id_str = clean_id_str(id_str)
    if re.search(TIMESTAMP_REGEX, c_id_str):
        name_time_trial = re.split(TIMESTAMP_REGEX, c_id_str)
        name_time_trial.pop()
        experiment_id = ''.join(name_time_trial)
        return experiment_id
    else:
        return None


def parse_experiment_name(id_str):
    c_id_str = clean_id_str(id_str)
    experiment_id = parse_experiment_id(c_id_str)
    if experiment_id is None:
        experiment_name = c_id_str
    else:
        experiment_name = re.sub(TIMESTAMP_REGEX, '', experiment_id).strip('-')
    assert experiment_name in EXPERIMENT_SPECS, \
        '{} not in EXPERIMENT_SPECS'.format(experiment_name)
    return experiment_name


def load_data_from_trial_id(id_str):
    experiment_id = parse_experiment_id(id_str)
    trial_id = parse_trial_id(id_str)
    data_filename = './data/{}/{}.json'.format(experiment_id, trial_id)
    try:
        data = json.loads(open(data_filename).read())
    except (FileNotFoundError, json.JSONDecodeError):
        data = None
    return data


def load_data_array_from_experiment_id(id_str):
    # to load all ./data files for a series of trials
    experiment_id = parse_experiment_id(id_str)
    data_path = './data/{}'.format(experiment_id)
    trial_id_array = [
        f for f in listdir(data_path)
        if (path.isfile(path.join(data_path, f)) and
            f.startswith(experiment_id) and
            f.endswith('.json'))
    ]
    return list(filter(None, [load_data_from_trial_id(trial_id)
                              for trial_id in trial_id_array]))


def save_experiment_data(data_df, trial_id):
    experiment_id = parse_experiment_id(trial_id)
    filename = './data/{0}/{0}_analysis_data.csv'.format(experiment_id)
    data_df.to_csv(filename, index=False)
    logger.info(
        'experiment data saved to {}'.format(filename))


def configure_hardware(RAND_SEED):
    '''configure rand seed, GPU'''
    from keras import backend as K
    if K.backend() == 'tensorflow':
        K.tf.set_random_seed(RAND_SEED)
    else:
        K.theano.tensor.shared_randomstreams.RandomStreams(seed=RAND_SEED)

    if K.backend() != 'tensorflow':
        # GPU config for tf only
        return

    process_num = PARALLEL_PROCESS_NUM if args.param_selection else 1
    tf = K.tf
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=1./float(process_num))
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)
    return sess


def debug_mem_usage():
    import psutil
    from mem_top import mem_top
    pid = getpid()
    logger.debug(
        'MEM USAGE for PID {}, MEM_INFO: {}\n{}'.format(
            pid, psutil.Process().memory_info(), mem_top()))


def del_self_attr(subject):
    self_attrs = list(subject.__dict__.keys())
    for attr in self_attrs:
        delattr(subject, attr)
    import gc
    gc.collect()
