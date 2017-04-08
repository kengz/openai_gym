import json
import numpy as np
import pandas as pd
import platform
import warnings
from os import environ
from rl.util import *

warnings.filterwarnings("ignore", module="matplotlib")

MPL_BACKEND = 'agg' if environ.get('CI') else (
    'macosx' if platform.system() == 'Darwin' else 'TkAgg')

STATS_COLS = [
    'best_session_epi',
    'best_session_id',
    'best_session_mean_rewards',
    'best_session_stability',
    'fitness_score',
    'mean_rewards_per_epi_stats_mean',
    'mean_rewards_stats_mean',
    'mean_rewards_stats_max',
    'epi_stats_mean',
    'epi_stats_min',
    'solved_ratio_of_sessions',
    'max_total_rewards_stats_mean',
    'trial_id',
]

EXPERIMENT_DATA_Y_COLS = [
    'fitness_score',
    'mean_rewards_stats_max',
    'max_total_rewards_stats_mean',
    'epi_stats_min',
]


# import matplotlib scoped to the class for gc in multiprocessing
def scoped_mpl_import():
    import matplotlib
    matplotlib.rcParams['backend'] = MPL_BACKEND

    import matplotlib.pyplot as plt
    plt.rcParams['toolbar'] = 'None'  # mute matplotlib toolbar

    import seaborn as sns
    sns.set(style="whitegrid", color_codes=True, font_scale=1.0,
            rc={'lines.linewidth': 1.0,
                'backend': matplotlib.rcParams['backend']})
    palette = sns.color_palette("Blues_d")
    palette.reverse()
    sns.set_palette(palette)

    return (matplotlib, plt, sns)


class Grapher(object):

    '''
    Grapher object that belongs to a Session
    to draw graphs from its data
    '''

    def __init__(self, session):
        if environ.get('CI'):
            return
        (_mpl, self.plt, _sns) = scoped_mpl_import()
        self.session = session
        self.graph_filename = self.session.graph_filename
        self.subgraphs = {}
        self.figure = self.plt.figure(facecolor='white', figsize=(8, 9))
        self.figure.suptitle(wrap_text(self.session.session_id))
        self.init_figure()

    def init_figure(self):
        # graph 1
        ax1 = self.figure.add_subplot(
            311,
            frame_on=False,
            title="\n\ntotal rewards per episode",
            ylabel='total rewards')
        p1, = ax1.plot([], [])
        self.subgraphs['total rewards'] = (ax1, p1)

        ax1e = ax1.twinx()
        ax1e.set_ylabel('exploration rate').set_color('r')
        ax1e.set_frame_on(False)
        ax1e.grid(False)
        p1e, = ax1e.plot([], [], 'r')
        self.subgraphs['e'] = (ax1e, p1e)

        # graph 2
        ax2 = self.figure.add_subplot(
            312,
            frame_on=False,
            title='mean rewards over last 100 episodes',
            ylabel='mean rewards')
        p2, = ax2.plot([], [], 'g')
        self.subgraphs['mean rewards'] = (ax2, p2)

        # graph 3
        ax3 = self.figure.add_subplot(
            313,
            frame_on=False,
            title='loss over time, episode',
            ylabel='loss')
        p3, = ax3.plot([], [])
        self.subgraphs['loss'] = (ax3, p3)

        self.plt.tight_layout()  # auto-fix spacing
        self.plt.ion()  # for live plot

    def plot(self):
        '''do live plotting'''
        if environ.get('CI'):
            return
        sys_vars = self.session.sys_vars
        ax1, p1 = self.subgraphs['total rewards']
        p1.set_ydata(sys_vars['total_rewards_history'])
        p1.set_xdata(np.arange(len(p1.get_ydata())))
        ax1.relim()
        ax1.autoscale_view(tight=True, scalex=True, scaley=True)

        ax1e, p1e = self.subgraphs['e']
        p1e.set_ydata(sys_vars['explore_history'])
        p1e.set_xdata(np.arange(len(p1e.get_ydata())))
        ax1e.relim()
        ax1e.autoscale_view(tight=True, scalex=True, scaley=True)

        ax2, p2 = self.subgraphs['mean rewards']
        p2.set_ydata(sys_vars['mean_rewards_history'])
        p2.set_xdata(np.arange(len(p2.get_ydata())))
        ax2.relim()
        ax2.autoscale_view(tight=True, scalex=True, scaley=True)

        ax3, p3 = self.subgraphs['loss']
        p3.set_ydata(sys_vars['loss'])
        p3.set_xdata(np.arange(len(p3.get_ydata())))
        ax3.relim()
        ax3.autoscale_view(tight=True, scalex=True, scaley=True)

        if sys_vars['RENDER']:
            self.plt.pause(0.01)
            self.plt.draw()
        self.save()
        import gc
        gc.collect()

    def save(self):
        '''save graph to filename'''
        self.figure.savefig(self.graph_filename)

    def clear(self):
        if environ.get('CI'):
            return
        self.plt.close()
        del_self_attr(self)


def calc_stability(sys_vars):
    '''
    calculate the stability of a session using its sys_vars
    when problem is unsolved (unbounded), use 1 sigma 95% of max
    stability 1 = perfectly stable
    0.5 = half-ish unstable
    0 = totally unstable, cannot yield solution
    '''
    total_r_history = sys_vars['total_rewards_history']
    if sys_vars['SOLVED_MEAN_REWARD'] is None:
        min_rewards = min(total_r_history)
        max_rewards = max(total_r_history)
        rewards_gap = max_rewards - min_rewards
        r_threshold = max_rewards - (0.10 * rewards_gap)
    else:
        r_threshold = sys_vars['SOLVED_MEAN_REWARD']
    # find index i.e. epi of first solved
    first_solved_epi = next(
        (idx for idx, total_r in enumerate(total_r_history)
            if total_r > r_threshold), None)
    last_epi = sys_vars['epi']
    stable_epi_count = len([
        total_r for total_r in total_r_history if total_r > r_threshold])

    if (first_solved_epi is None) or (last_epi == first_solved_epi):
        mastery_gap = np.inf
    else:  # get max if mastery_gap is smaller (faster) than needed - perfect
        mastery_gap = last_epi - first_solved_epi
    stability = stable_epi_count / mastery_gap
    return stability


def fitness_score(stats):
    '''
    calculate the fitness score (see doc Metrics for more)
    1. solution rewards
    2. solving speed: /epi
    3. stability
    4. consistency
    5. granularity
    6. amplification of good results
    7. distinguishability
    '''
    mean_rewards_per_epi = stats['mean_rewards_per_epi_stats']['mean']
    stability = stats['stability_stats']['mean']
    consistency = stats['solved_ratio_of_sessions']
    amplifier = (1+stability)*((1+consistency)**2)
    distinguisher = amplifier ** np.sign(mean_rewards_per_epi)
    fitness = mean_rewards_per_epi * distinguisher
    return fitness


def ideal_fitness_score(problem):
    '''
    calculate the ideal fitness_score with perfect solved ratio
    for hyperparameter optimization to select
    '''
    if problem['SOLVED_MEAN_REWARD'] is None:
        return np.inf  # for unsolved environments
    solved_mean_reward = problem['SOLVED_MEAN_REWARD']
    max_episodes = problem['MAX_EPISODES']
    solved_epi_speedup = 3
    ideal_epi = max_episodes / solved_epi_speedup
    ideal_mean_rewards_per_epi = solved_mean_reward / ideal_epi
    ideal_stability = 1
    ideal_consistency = 1
    amplifier = (1+ideal_stability)*((1+ideal_consistency)**2)
    distinguisher = amplifier ** np.sign(ideal_mean_rewards_per_epi)
    ideal_fitness = ideal_mean_rewards_per_epi * distinguisher
    return ideal_fitness


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


def compose_data(trial):
    '''
    compose raw data from an trial object
    into useful summary and full metrics for analysis
    '''
    sys_vars_array = trial.data['sys_vars_array']

    # collect all data from sys_vars_array
    solved_sys_vars_array = list(filter(
        lambda sv: sv['solved'], sys_vars_array))
    errored_array = list(map(
        lambda sv: sv['errored'], sys_vars_array))
    mean_rewards_array = np.array(list(map(
        lambda sv: sv['mean_rewards'], sys_vars_array)))
    max_total_rewards_array = np.array(list(map(
        lambda sv: np.max(sv['total_rewards_history']), sys_vars_array)))
    epi_array = np.array(list(map(lambda sv: sv['epi'], sys_vars_array)))
    mean_rewards_per_epi_array = np.divide(mean_rewards_array, epi_array + 1)
    stability_array = list(map(calc_stability, sys_vars_array))
    t_array = np.array(list(map(lambda sv: sv['t'], sys_vars_array)))
    time_taken_array = np.array(list(map(
        lambda sv: timestamp_elapse_to_seconds(sv['time_taken']),
        sys_vars_array)))
    solved_epi_array = np.array(list(map(
        lambda sv: sv['epi'], solved_sys_vars_array)))
    solved_t_array = np.array(list(map(
        lambda sv: sv['t'], solved_sys_vars_array)))
    solved_time_taken_array = np.array(list(map(
        lambda sv: timestamp_elapse_to_seconds(sv['time_taken']),
        solved_sys_vars_array)))
    best_idx = list(mean_rewards_per_epi_array).index(
        max(mean_rewards_per_epi_array))
    best_session_id = '{}_s{}'.format(trial.data['trial_id'], best_idx)

    # compose sys_vars stats
    stats = {
        'best_session_epi': epi_array.tolist()[best_idx],
        'best_session_id': best_session_id,
        'best_session_mean_rewards': mean_rewards_array[best_idx],
        'best_session_stability': stability_array[best_idx],
        'errored': any(errored_array),
        'epi_stats': basic_stats(epi_array),
        'max_total_rewards_stats': basic_stats(max_total_rewards_array),
        'mean_rewards_stats': basic_stats(mean_rewards_array),
        'mean_rewards_per_epi_stats': basic_stats(
            mean_rewards_per_epi_array),
        'num_of_sessions': len(sys_vars_array),
        'solved_epi_stats': basic_stats(solved_epi_array),
        'solved_num_of_sessions': len(solved_sys_vars_array),
        'solved_ratio_of_sessions': float(len(
            solved_sys_vars_array)) / trial.times,
        'solved_t_stats': basic_stats(solved_t_array),
        'solved_time_taken_stats': basic_stats(solved_time_taken_array),
        'stability_stats': basic_stats(stability_array),
        't_stats': basic_stats(t_array),
        'time_taken_stats': basic_stats(time_taken_array),
    }
    stats.update({
        'fitness_score': fitness_score(stats)
    })

    # summary metrics picked from stats
    metrics = {
        'best_session_epi': stats['best_session_epi'],
        'best_session_id': stats['best_session_id'],
        'best_session_mean_rewards': stats['best_session_mean_rewards'],
        'best_session_stability': stats['best_session_stability'],
        'fitness_score': stats['fitness_score'],
        'mean_rewards_per_epi_stats_mean': stats[
            'mean_rewards_per_epi_stats']['mean'],
        'solved_ratio_of_sessions': stats['solved_ratio_of_sessions'],
        't_stats_mean': stats['t_stats']['mean'],
    }

    # param variables for independent vars of trials
    param_variables = {
        pv: trial.experiment_spec['param'][pv] for
        pv in trial.param_variables}

    trial.data['metrics'].update(metrics)
    trial.data['param_variables'] = param_variables
    trial.data['stats'] = stats
    return trial.data


# order a unique df categorical data for plotting
def order_category(uniq_df):
    uniq_list = list(uniq_df)
    try:
        uniq_dict = {k: json.loads(k) for k in uniq_list}
        sorted_pair = sorted(uniq_dict.items(), key=lambda x: x[1])
        return [pair[0] for pair in sorted_pair]
    except (json.JSONDecodeError, TypeError):
        return list(sorted(uniq_list))


# plot the experiment data from data_df
# X are columns with name starting with 'variable_'
# Y cols are defined below
def plot_experiment(data_df, trial_id):
    if len(data_df) < 2:  # no multi selection
        return
    (_mpl, _plt, sns) = scoped_mpl_import()
    experiment_id = parse_experiment_id(trial_id)
    hue = 'solved_ratio_of_sessions'
    data_df = data_df.sort_values(hue)
    fitness_hue = 'fitness_score_bin'
    data_df[fitness_hue] = pd.cut(data_df['fitness_score'], bins=5)
    X_cols = list(filter(lambda c: c.startswith('variable_'), data_df.columns))
    col_size = len(X_cols)
    row_size = len(EXPERIMENT_DATA_Y_COLS)
    groups = data_df.groupby(hue)

    # for main grid plot
    sns_only = True
    big_fig, axes = sns.plt.subplots(
        row_size, col_size, figsize=(col_size*4, row_size*3),
        sharex='col', sharey='row')
    for ix, x in enumerate(X_cols):
        for iy, y in enumerate(EXPERIMENT_DATA_Y_COLS):
            big_ax = axes[iy] if col_size == 1 else axes[iy][ix]
            uniq_df = data_df[x].unique()
            if (data_df[x].dtype.name == 'category' or
                    len(uniq_df) <= 5):
                order = order_category(uniq_df)
                sns.swarmplot(
                    data=data_df, x=x, y=y, hue=hue, size=3,
                    order=order, ax=big_ax)
            else:
                sns_only = False
                big_ax.margins(0.05)
                big_ax.xaxis.grid(False)
                for _, group in groups:
                    big_ax.plot(group[x], group[y], label=hue,
                                marker='o', ms=3, linestyle='')
                    big_ax.set_xlabel(x)
                    big_ax.set_ylabel(y)

            big_ax.legend_ = None  # set common legend below
            # label only left and bottom axes
            if iy != row_size - 1:
                big_ax.set_xlabel('')
            if ix != 0:
                big_ax.set_ylabel('')

    big_fig.tight_layout()
    big_fig.suptitle(wrap_text(experiment_id))
    legend_labels = None if sns_only else sorted(data_df[hue].unique())
    legend_ms = 0.5 if sns_only else 1
    legend = sns.plt.legend(title='solved_ratio_of_sessions',
                            labels=legend_labels, markerscale=legend_ms,
                            fontsize=10, loc='center right',
                            bbox_to_anchor=(1.1+col_size*0.1, row_size+0.1))
    legend.get_title().set_fontsize('10')
    big_fig.subplots_adjust(top=0.96, right=0.9)

    filename = './data/{0}/{0}_analysis.png'.format(
        experiment_id)
    big_fig.savefig(filename)
    big_fig.clear()

    # use numerical, since contour only makes sense for ordered azes
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_X_cols = list(
        filter(lambda x: data_df[x].dtype in numerics, X_cols))
    with sns.axes_style('white', {'axes.linewidth': 0.2}):
        g = sns.pairplot(
            data_df, vars=numeric_X_cols, hue=fitness_hue,
            size=3, aspect=1, plot_kws={'s': 50, 'alpha': 0.5})
        g.fig.suptitle(wrap_text(experiment_id))
        g = g.add_legend()
        filename = './data/{0}/{0}_analysis_correlation.png'.format(
            experiment_id)
        g.savefig(filename)
        g.fig.clear()

    sns.plt.close()


def analyze_data(experiment_data_or_experiment_id):
    '''
    get all the data from all trials.run()
    or read from all data files matching the prefix of trial_id
    e.g. usage without running:
    experiment_id = 'DevCartPole-v0_DQN_LinearMemoryWithForgetting_BoltzmannPolicy_2017-01-15_142810'
    analyze_data(experiment_id)
    '''
    if isinstance(experiment_data_or_experiment_id, str):
        experiment_data = load_data_array_from_experiment_id(
            experiment_data_or_experiment_id)
    else:
        experiment_data = experiment_data_or_experiment_id

    stats_array, param_variables_array = [], []
    for data in experiment_data:
        stats = flatten_dict(data['stats'])
        stats.update({'trial_id': data['trial_id']})
        param_variables = flat_cast_dict(data['param_variables'])
        if stats['errored']:  # remove errored trials
            continue
        stats_array.append(stats)
        param_variables_array.append(param_variables)

    raw_stats_df = pd.DataFrame.from_dict(stats_array)
    stats_df = raw_stats_df[STATS_COLS]

    param_variables_df = pd.DataFrame.from_dict(param_variables_array)
    param_variables_df.columns = [
        'variable_'+c for c in param_variables_df.columns]

    data_df = pd.concat([stats_df, param_variables_df], axis=1)
    for c in data_df.columns:
        if data_df[c].dtype == object:  # guard
            data_df[c] = data_df[c].astype('category')

    data_df.sort_values(
        ['fitness_score'], inplace=True, ascending=False)

    trial_id = experiment_data[0]['trial_id']
    save_experiment_data(data_df, trial_id)
    plot_experiment(data_df, trial_id)
    return data_df
