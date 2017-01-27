import argparse
import collections
import copy
import itertools
import json
import logging
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from os import path, environ
from textwrap import wrap


# parse_args to add flag
parser = argparse.ArgumentParser(description='Set flag for functions')
parser.add_argument("-d", "--debug",
                    help="activate debug log",
                    action="store_const",
                    dest="loglevel",
                    const=logging.DEBUG,
                    default=logging.INFO)
parser.add_argument("-b", "--blind",
                    help="dont render graphics",
                    action="store_const",
                    dest="render",
                    const=False,
                    default=True)
parser.add_argument("-s", "--sess",
                    help="specifies session to run, see sess_specs.json",
                    action="store",
                    type=str,
                    nargs='?',
                    dest="sess_name",
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
parser.add_argument("-l", "--line_search",
                    help="run line search instead of grid search if present",
                    action="store_true",
                    dest="line_search",
                    default=False)
parser.add_argument("-g", "--graph",
                    help="plot metrics graphs live",
                    action="store_true",
                    dest="plot_graph",
                    default=False)
args = parser.parse_args([]) if environ.get('CI') else parser.parse_args()

# Goddam python logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.setLevel(args.loglevel)
logger.addHandler(handler)
logger.propagate = False


def log_self(subject):
    logger.info('{}, params: {}'.format(
        subject.__class__.__name__,
        to_json(subject.__dict__)))


def wrap_text(text):
    return '\n'.join(wrap(text, 60))


def print_line(line='-'):
    if environ.get('CI'):
        return
    _rows, columns = os.popen('stty size', 'r').read().split()
    line_str = line*int(columns)
    print(line_str)


def log_delimiter(msg, line='-'):
    print('{:\n>3}'.format(''))
    print_line(line)
    print(msg)
    print_line(line)
    print('{:\n>3}'.format(''))


def timestamp():
    '''timestamp used for filename'''
    return '{:%Y-%m-%d_%H%M%S}'.format(datetime.now())


def timestamp_elapse(s1, s2):
    '''calculate the time elapsed between timestamps from s1 to s2'''
    FMT = '%Y-%m-%d_%H%M%S'
    delta_t = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return str(delta_t)


def timestamp_elapse_to_seconds(s1):
    a = datetime.strptime(s1, '%H:%M:%S')
    secs = timedelta(hours=a.hour, minutes=a.minute, seconds=a.second).seconds
    return secs


def basic_stats(array):
    '''generate the basic stats for a numerical array'''
    if not len(array):
        return None
    return {
        'min': np.min(array).astype(float),
        'max': np.max(array).astype(float),
        'mean': np.mean(array).astype(float),
        'std': np.std(array).astype(float),
    }


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
    else:
        raise TypeError(
            "Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def format_obj_dict(obj, keys):
    if isinstance(obj, dict):
        return to_json(
            {k: obj.get(k) for k in keys if obj.get(k) is not None})
    else:
        return to_json(
            {k: getattr(obj, k, None) for k in keys
             if getattr(obj, k, None) is not None})


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


# convert a dict of param ranges into
# a list parameter settings corresponding
# to a line search of the param range
# for each param
# All other parameters set to default vals
def param_line_search(sess_spec):
    default_param = sess_spec['param']
    param_range = sess_spec['param_range']
    keys = param_range.keys()
    param_list = []
    for key in keys:
        vals = param_range[key]
        for val in vals:
            param = copy.deepcopy(default_param)
            param[key] = val
            param_list.append(param)
    return param_list


# convert a dict of param ranges into
# a list of cartesian products of param_range
# e.g. {'a': [1,2], 'b': [3]} into
# [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
def param_product(sess_spec):
    default_param = sess_spec['param']
    param_range = sess_spec['param_range']
    keys = param_range.keys()
    range_vals = param_range.values()
    param_grid = []
    for vals in itertools.product(*range_vals):
        param = copy.deepcopy(default_param)
        param.update(dict(zip(keys, vals)))
        param_grid.append(param)
    return param_grid


# for param selection
def generate_sess_spec_grid(sess_spec, param_grid):
    sess_spec_grid = [{
        'problem': sess_spec['problem'],
        'Agent': sess_spec['Agent'],
        'Memory': sess_spec['Memory'],
        'Policy': sess_spec['Policy'],
        'PreProcessor': sess_spec['PreProcessor'],
        'param': param,
    } for param in param_grid]
    return sess_spec_grid


# helper wrapper for multiprocessing
def mp_run_helper(experiment):
    return experiment.run()


def prefix_id_from_experiment_id(experiment_id):
    str_arr = experiment_id.split('_')
    if str_arr[-1].startswith('e'):
        str_arr.pop()
    return '_'.join(str_arr)


def load_data_from_experiment_id(experiment_id):
    experiment_id = experiment_id.split(
        '/').pop().split('.').pop(0)
    prefix_id = prefix_id_from_experiment_id(experiment_id)
    data_filename = './data/{}/{}.json'.format(prefix_id, experiment_id)
    data = json.loads(open(data_filename).read())
    return data


def load_data_array_from_prefix_id(prefix_id):
    # to load all ./data files for a series of experiments
    prefix_id = prefix_id_from_experiment_id(prefix_id)
    data_path = './data/{}'.format(prefix_id)
    experiment_id_array = [
        f for f in os.listdir(data_path)
        if (path.isfile(path.join(data_path, f)) and
            f.startswith(prefix_id) and
            f.endswith('.json'))
    ]
    return [load_data_from_experiment_id(experiment_id)
            for experiment_id in experiment_id_array]
