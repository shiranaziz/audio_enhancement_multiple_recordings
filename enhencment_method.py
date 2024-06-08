import librosa
import pandas as pd
import os
import torch
import soundfile as sf
from torchmetrics import ScaleInvariantSignalNoiseRatio
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from create_dataset import DataSet, ignor_ds_store, create_data_folders
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import pyrallis
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pesq import pesq as pesq_function
from pystoi import stoi
import scipy.stats


DEFAULT_SAMPLE_RATE = 16000


class DataSet(Enum):
    SOURCE_MUSIC_NOISE_SPEECH = "source_music_noise_speech"
    SOURCE_MUSIC_NOISE_AUDIOSET = "source_music_noise_audioset"
    SOURCE_MUSIC_NOISE_DEMAND = "source_music_noise_demand"
    SOURCE_SPEECH_NOISE_SPEECH = "source_speech_noise_speech"
    SOURCE_SPEECH_NOISE_AUDIOSET = "source_speech_noise_audioset"
    SOURCE_SPEECH_NOISE_DEMAND = "source_speech_noise_demand"
    SOURCE_SPEECH_NOISE_SPEECH_PACKETLOSS = "source_speech_noise_speech_packetloos"
    SOURCE_SPEECH_NOISE_DEMAND_PACKETLOSS = "source_speech_noise_demand_packetloos"
    SOURCE_MUSIC_NOISE_SPEECH_PACKETLOSS = "source_music_noise_speech_packetloos"
    SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES = "source_music_noise_speech_changed_number_of_noises"


class Methods(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX_ELIMINATION = "max_elimination"
    OURS = "ours"


class Metrics(Enum):
    PESQ = "pesq"
    SI_SNR = "si-snr"
    STOI = "stoi"


@dataclass
class EnhencmentMethodConfig:
    dataset_type: DataSet
    dataset_path: str
    output_dataset_path: str
    method_to_examine: List[Methods]
    metrics_to_examine: List[Metrics]
    predefined_samplerate: int = 16000


def si_snr(preds, target):
    si_snr = ScaleInvariantSignalNoiseRatio()
    return si_snr(torch.from_numpy(preds), torch.from_numpy(target)).item()


def pesq(target, preds):
    try:
        return pesq_function(DEFAULT_SAMPLE_RATE, target, preds, mode="wb")
    except Exception as e:
        return 0.0


def stoi_func(target, preds):
    try:
        return stoi(target, preds, DEFAULT_SAMPLE_RATE, extended=False)
    except:
        return 0.0


def evaluation(clean_data, method_signal, example, snr, method, rows, cfg):
    x = clean_data[:method_signal.shape[0]]
    res = 0
    for metric in cfg.metrics_to_examine:
        if metric == Metrics.SI_SNR:
            res = si_snr(method_signal, x)
        if metric == Metrics.PESQ:
            res = pesq(method_signal, x)
        if metric == Metrics.STOI:
            res = stoi_func(method_signal, x)
        rows.append(
            {'example_name': example, 'Input SNR': snr, 'alg': method, 'Output eval': res, 'metric': metric.value})
    return rows


def baseline_mean(all_files_stft, output_path):
    mean_STFT = np.mean(list(all_files_stft.values()), axis=0)
    wav_optimize = librosa.istft(mean_STFT)
    wav_optimize = wav_optimize
    write(output_path + 'baseline_mean_spec.wav', DEFAULT_SAMPLE_RATE, wav_optimize)
    return wav_optimize


def baseline_median(all_files_stft, output_path):
    mean_STFT = np.mean(list(all_files_stft.values()), axis=0)
    median_amplitude = np.median(np.abs(list(all_files_stft.values())), axis=0)
    phase = np.angle(mean_STFT)
    signal = median_amplitude * np.exp(1j * phase)
    wav_optimize = librosa.istft(signal)
    wav_optimize = wav_optimize
    write(output_path + 'baseline_median_spec.wav', DEFAULT_SAMPLE_RATE, wav_optimize)
    return wav_optimize


def max_elimination(all_files_stft, output_path):
    stacked_stft = []
    for id, stft in all_files_stft.items():
        stacked_stft.append(stft)

    stacked_stft = np.stack(stacked_stft)
    stacked_amp = np.abs(stacked_stft)
    stacked_mask = (stacked_amp.max(axis=0, keepdims=True) == stacked_amp).astype('int')
    stacked_mask = 1 - stacked_mask
    stacked_res = np.multiply(stacked_stft, stacked_mask)

    signal = stacked_res.sum(axis=0) / (len(list(all_files_stft.values())) - 1)

    wav_optimize = librosa.istft(signal)
    write(output_path + 'max_elimination.wav', DEFAULT_SAMPLE_RATE, wav_optimize)

    return wav_optimize


def spectogram_median(spectogram_dict):
    amplitude = {}
    for id, spectogram in spectogram_dict.items():
        spectogram_amplitude = np.abs(spectogram)
        amplitude[id] = spectogram_amplitude
    return np.median(list(amplitude.values()), axis=0)


def spectogram_histerises(spectogram_dict, file_chosen, comparison_factor, first_amplitude_factor,
                          second_amplitude_factor, lower_bound, amplitude_resize):
    reference_spectogram = spectogram_dict[file_chosen]
    reference_amplitude = np.abs(reference_spectogram)

    last_iter_spec = reference_spectogram
    cur_iter_spec = np.where((reference_amplitude > first_amplitude_factor * comparison_factor)
                             | (reference_amplitude < lower_bound * comparison_factor),
                             amplitude_resize * reference_spectogram, reference_spectogram)
    indecator_spec = (reference_amplitude > first_amplitude_factor * comparison_factor) | \
                     (reference_amplitude < lower_bound * comparison_factor)

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    i = 0
    while (np.sum(last_iter_spec != cur_iter_spec) > 0):
        # print(i)
        last_iter_spec = cur_iter_spec
        changed_neighbor = signal.convolve2d(indecator_spec, kernel)[1:indecator_spec.shape[0] + 1,
                           1:indecator_spec.shape[1] + 1]
        cur_iter_spec = np.where((((reference_amplitude > second_amplitude_factor * comparison_factor) |
                                   (reference_amplitude < lower_bound * comparison_factor))
                                  & changed_neighbor) | indecator_spec, amplitude_resize * reference_spectogram,
                                 reference_spectogram)
        indecator_spec = (((reference_amplitude > second_amplitude_factor * comparison_factor) | (
                reference_amplitude < lower_bound * comparison_factor))
                          & changed_neighbor) | indecator_spec

        i += 1
    return cur_iter_spec, indecator_spec


def weighted_mean_signals(all_files_stft, output_path, first_amplitude_factor, second_amplitude_factor, lower_bound,
                          amplitude_resize=0):
    cumulative_STFT = np.zeros(list(all_files_stft.values())[0].shape)
    cumulative_STFT_wo_lower = np.zeros_like(cumulative_STFT)
    cumulative_indicator_spectogram = np.zeros(list(all_files_stft.values())[0].shape)
    cumulative_indicator_spectogram_wo_lower = np.zeros_like(cumulative_indicator_spectogram)
    comparison_factor = spectogram_median(all_files_stft)
    for id in all_files_stft.keys():
        reference_file = id
        res_optimize, res_black_white = spectogram_histerises(all_files_stft, reference_file, comparison_factor,
                                                              first_amplitude_factor, second_amplitude_factor,
                                                              lower_bound, amplitude_resize=0)
        res_optimize_wo_lower, res_black_white_wo_lower = spectogram_histerises(all_files_stft, reference_file,
                                                                                comparison_factor,
                                                                                first_amplitude_factor,
                                                                                second_amplitude_factor, 0,
                                                                                amplitude_resize=0)
        cumulative_STFT = cumulative_STFT + res_optimize
        cumulative_STFT_wo_lower = cumulative_STFT_wo_lower + res_optimize_wo_lower
        cumulative_indicator_spectogram = cumulative_indicator_spectogram + res_black_white
        cumulative_indicator_spectogram_wo_lower = cumulative_indicator_spectogram_wo_lower + res_black_white_wo_lower

    num_files = len(list(all_files_stft.values()))
    cumulative_indicator_filter = num_files - cumulative_indicator_spectogram
    mean_STFT = cumulative_STFT / cumulative_indicator_filter

    unmarked_ind = np.argwhere(cumulative_indicator_filter == 0)
    # print("num files:", num_files)
    # print("num zeros:", len(unmarked_ind))
    if len(unmarked_ind) > 0:
        mean_STFT[unmarked_ind[:, 0], unmarked_ind[:, 1]] = cumulative_STFT_wo_lower[
                                                                unmarked_ind[:, 0], unmarked_ind[:, 1]] / \
                                                            cumulative_indicator_spectogram_wo_lower[
                                                                unmarked_ind[:, 0], unmarked_ind[:, 1]]
        # print('fixing div zero')

    wav_optimize = librosa.istft(mean_STFT)
    write(output_path + 'ours.wav', DEFAULT_SAMPLE_RATE, wav_optimize)

    return wav_optimize


def run_enhancement_method(input_path, output_path, clean_data_path, rows, snr, example, cfg):
    all_files_sample_rate = {}
    all_files_data = {}
    all_files_stft = {}

    dir = os.listdir(input_path)
    for i, vid_path in enumerate(dir):
        id = vid_path.split(".wav")[0]

        data, samplerate = sf.read(input_path + vid_path, dtype='float32')

        all_files_sample_rate[id] = samplerate
        all_files_data[id] = data
        spectogram = librosa.stft(data)

        all_files_stft[id] = spectogram

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    clean_data, samplerate = sf.read(clean_data_path, dtype='float32')
    for method in cfg.method_to_examine:
        pred = 0
        if method == Methods.MEAN:
            pred = baseline_mean(all_files_stft, output_path)
        if method == Methods.MEDIAN:
            pred = baseline_median(all_files_stft, output_path)
        if method == Methods.MAX_ELIMINATION:
            pred = max_elimination(all_files_stft, output_path)
        if method == Methods.OURS:
            pred = weighted_mean_signals(all_files_stft, output_path, 1.15, 1.1, 0.01)
        rows = evaluation(clean_data, pred, example, snr, method.value, rows, cfg)
    return rows


def run_single_example(input_path, output_path, rows, snr, example, cfg):
    data_path = f'{input_path}/signal+noise/'
    output_path = f'{output_path}/'
    clean_data_path = f'{input_path}/clean_signal/'
    clean_data_path_dir = ignor_ds_store(os.listdir(clean_data_path))
    rows = run_enhancement_method(data_path, output_path, f'{clean_data_path}{clean_data_path_dir[0]}', rows, snr,
                                  example, cfg)
    return rows


def run_multiple_examples(input_path, output_path, csv_path, cfg):
    snr_input_path_dir = ignor_ds_store(os.listdir(input_path))
    rows = []

    for snr in tqdm(snr_input_path_dir):
        # print(snr)
        if not os.path.isdir(f'{output_path}{snr}'):
            os.mkdir(f'{output_path}{snr}')
        s = int(snr.split("_")[1])
        examples_path_dir = ignor_ds_store(os.listdir(f'{input_path}{snr}'))

        for example in examples_path_dir:
            # print(example)
            if not os.path.isdir(f'{output_path}{snr}/{example}'):
                os.mkdir(f'{output_path}{snr}/{example}')
            rows = run_single_example(f'{input_path}{snr}/{example}', f'{output_path}{snr}/{example}', rows, s, example,
                                      cfg)
    res_df = pd.DataFrame(rows)
    res_df.to_csv(f'{csv_path}{cfg.dataset_type.value}.csv')


def run_enhancement_method_multi_m(input_path, output_path, clean_data_path, rows, example, s):
    clean_data, samplerate = sf.read(clean_data_path, dtype='float32')

    all_files_sample_rate = {}
    all_files_data = {}
    all_files_stft = {}

    dir = os.listdir(input_path)
    microphones = np.random.choice(np.arange(0, len(dir)), size=10, replace=False)

    for m in microphones[:3]:
        id = dir[m].split(".wav")[0]
        data, samplerate = sf.read(f'{input_path}{id}.wav', dtype='float32')
        all_files_sample_rate[id] = samplerate
        all_files_data[id] = data
        all_files_stft[id] = librosa.stft(data)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(f'{output_path}{3}'):
        os.mkdir(f'{output_path}{3}')

    max_elimination_signal = max_elimination(all_files_stft, f'{output_path}{3}/')
    x = clean_data[:max_elimination_signal.shape[0]]
    max_elimination_res = si_snr(max_elimination_signal, x)

    our_method_signal = weighted_mean_signals(all_files_stft, f'{output_path}{3}/', 1.15, 1.1, 0.01)
    x = clean_data[:our_method_signal.shape[0]]
    our_method_res = si_snr(our_method_signal, x)

    rows.append({'example_name': example, 'Input SNR': s, 'Number of mics': 3, 'alg': 'Maximum Component Elimination',
                 'Output SI-SNR': max_elimination_res})
    rows.append({'example_name': example, 'Input SNR': s, 'Number of mics': 3, 'alg': 'our with hysteresis',
                 'Output SI-SNR': our_method_res})

    for j in range(3, 10):
        id = dir[j].split(".wav")[0]
        data, samplerate = sf.read(f'{input_path}{id}.wav', dtype='float32')
        all_files_sample_rate[id] = samplerate
        all_files_data[id] = data
        all_files_stft[id] = librosa.stft(data)
        if not os.path.isdir(f'{output_path}{j + 1}/'):
            os.mkdir(f'{output_path}{j + 1}/')

        new_method = max_elimination(all_files_stft, f'{output_path}/{j + 1}/')
        new_method_res = evaluation(clean_data, new_method)
        our_method_signal = weighted_mean_signals(all_files_stft, f'{output_path}{j + 1}/', 1.15, 1.1, 0.01)
        our_method_res = evaluation(clean_data, our_method_signal)

        rows.append(
            {'example_name': example, 'Input SNR': s, 'Number of mics': j + 1, 'alg': 'Maximum Component Elimination',
             'Output SI-SNR': new_method_res})
        rows.append({'example_name': example, 'Input SNR': s, 'Number of mics': j + 1, 'alg': 'our with hysteresis',
                     'Output SI-SNR': our_method_res})

    return


def run_multiple_examples_multi_m(input_path, output_path, csv_path, cfg):
    snr_input_path_dir = os.listdir(input_path)
    rows = []

    for snr in tqdm(snr_input_path_dir):
        if not os.path.isdir(f'{output_path}{snr}'):
            os.mkdir(f'{output_path}{snr}')
        s = int(snr.split("_")[1])
        examples_path_dir = os.listdir(f'{input_path}{snr}')
        for example in examples_path_dir:
            if not os.path.isdir(f'{output_path}{snr}/{example}'):
                os.mkdir(f'{output_path}{snr}/{example}')
            data_path = f'{input_path}{snr}/{example}/signal+noise/'
            output_path_snr = f'{output_path}{snr}/{example}/'
            clean_data_path = f'{input_path}{snr}/{example}/clean_signal/'
            run_enhancement_method_multi_m(data_path, output_path_snr,
                                           f'{clean_data_path}/{os.listdir(clean_data_path)[0]}', rows,
                                           example, s)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(f'{csv_path}{cfg.dataset_type.value}.csv')


def run_collect_df(cfg: EnhencmentMethodConfig):
    if not os.path.isdir(f'{cfg.output_dataset_path}csv'):
        os.mkdir(f'{cfg.output_dataset_path}csv')
    create_data_folders(cfg.output_dataset_path, 'methods_outputs', cfg.dataset_type.value)
    run_multiple_examples(f'{cfg.dataset_path}{cfg.dataset_type.value}/',
                          f'{cfg.output_dataset_path}methods_outputs/{cfg.dataset_type.value}/',
                          f'{cfg.output_dataset_path}csv/', cfg)
    print(f'finish {cfg.dataset_type}')


def run_collect_df_multi_m(cfg: EnhencmentMethodConfig):
    if not os.path.isdir(f'{cfg.output_dataset_path}csv'):
        os.mkdir(f'{cfg.output_dataset_path}csv')
    create_data_folders(cfg.output_dataset_path, 'methods_outputs', cfg.dataset_type.value)

    run_multiple_examples_multi_m(f'{cfg.dataset_path}{cfg.dataset_type.value}/',
                                  f'{cfg.output_dataset_path}methods_outputs/{cfg.dataset_type.value}/',
                                  f'{cfg.output_dataset_path}csv/', cfg)
    print(f'finish {cfg.dataset_type}')

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def read_csv_and_plot_graph(path, cfg):
    if not os.path.isdir(f'{cfg.output_dataset_path}plot'):
        os.mkdir(f'{cfg.output_dataset_path}plot')
    # res_df = pd.read_csv(path)
    # sns.set(font_scale=5)
    # plt.figure(figsize=(20, 20))
    # ax = sns.lineplot(data=res_df, x='Input SNR', y='Output SI-SNR', hue='alg',
    #                   err_kws={'capsize': 5, 'elinewidth': 3, 'capthick': 3}, err_style='bars', linewidth=5)
    # ax.set_ylim([-10, 32])
    # ax.set_xlabel('Input SNR', fontsize=60)
    # ax.set_ylabel('Output SI-SNR', fontsize=60)
    # leg = ax.legend(fontsize='50', loc='lower right')
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(5)
    # plt.savefig(f'{cfg.output_dataset_path}plot/{cfg.dataset_type}.png')
    # plt.show()

    result = pd.read_csv(path)
    for metric in cfg.metrics_to_examine:
        sns.set(font_scale=5)
        plt.figure(figsize=(20, 20))
        metric_df = result[result['metric'] == metric.value]
        ax = sns.lineplot(data=metric_df, x='Input SNR', y=f'Output eval', hue='alg',
                          hue_order=[Methods.OURS.value, Methods.MAX_ELIMINATION.value], err_kws={'capsize': 5, 'elinewidth': 3, 'capthick': 3},
                          err_style='bars', linewidth=5)

        ax.set_xlabel('Input SNR', fontsize=60)
        ax.set_ylabel(f'Output {metric.value}', fontsize=60)
        location = 'lower right'
        if metric == Metrics.STOI:
            location = 'lower right'
            ax.set_ylim([0, 1])
        if metric == Metrics.PESQ:
            location = 'upper left'
            ax.set_ylim([0, 5])
        if metric == Metrics.SI_SNR:
            location = 'lower right'
        leg = ax.legend(fontsize='50', loc=location)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5)
        plt.savefig(f'{cfg.output_dataset_path}plot/{metric.value}_{cfg.dataset_type.value}.png')
        # plt.show()
        # mean_df = result.groupby(['Input SNR', 'alg', metric.value]).agg(mean_xxx=(f'Output {metric}', mean_confidence_interval)).reset_index()
        # print(mean_df)


def read_csv_and_plot_graph_multi_m(path, cfg):
    res_df = pd.read_csv(path)
    res_df = res_df.drop((res_df[res_df['alg'] == 'Ours without ']).index)
    res_df = res_df.replace('Maximum Component Elimination', 'Max Elimination')
    res_df = res_df.replace('our with hysteresis', 'Ours')
    # f, ax = plt.subplots(1, 3, figsize=(22,10), sharey=True)

    for i, m in enumerate([3, 4, 5, 6, 7, 8, 9, 10]):
        df = res_df[res_df['Number of mics'] == m]
        plt.figure(figsize=(20, 20))
        sns.set(font_scale=5)

        sns.barplot(data=df, x='Input SNR', y='Output SI-SNR', hue='alg', hue_order=['Ours', 'Max Elimination'])

        # x.tick_params(axis='x', labelsize=30)
        # x.tick_params(axis='y', labelsize=30)
        # x.set_xlabel('Input SNR', size=60)
        # x.set_ylabel('Output SI-SNR', size=60)
        plt.ylim([0, 35])
        plt.legend(fontsize='50', loc='upper left')
        plt.savefig(f'{cfg.output_dataset_path}plot/{cfg.dataset_type.value}.png')
        plt.show()


@pyrallis.wrap()
def main(cfg: EnhencmentMethodConfig):
    if cfg.dataset_type == DataSet.SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES:
        run_collect_df_multi_m(cfg)
        read_csv_and_plot_graph_multi_m(f'{cfg.output_dataset_path}csv/{cfg.dataset_type.value}.csv', cfg)
    else:
        run_collect_df(cfg)
        read_csv_and_plot_graph(f'{cfg.output_dataset_path}csv/{cfg.dataset_type.value}.csv', cfg)


if __name__ == '__main__':
    main()
