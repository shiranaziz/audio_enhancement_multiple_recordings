from dataclasses import dataclass, field
from enum import Enum
from typing import List
import pyrallis

import numpy as np
import os
import soundfile as sf
from scipy.io import wavfile
import librosa
import shutil
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


# DEFAULT_SAMPLE_RATE = 16000

class DataSet(Enum):
    SOURCE_MUSIC_NOISE_SPEECH = "source_music_noise_speech"
    SOURCE_MUSIC_NOISE_AUDIOSET = "source_music_noise_audioset"
    SOURCE_MUSIC_NOISE_DEMAND = "source_music_noise_demand"
    SOURCE_SPEECH_NOISE_SPEECH = "source_speech_noise_speech"
    SOURCE_SPEECH_NOISE_AUDIOSET = "source_speech_noise_audioset"
    SOURCE_SPEECH_NOISE_DEMAND = "source_speech_noise_demand"
    SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES = "source_music_noise_speech_changed_number_of_noises"


# DEFAULT_SAMPLE_RATE = 16000
@dataclass
class CreateDatesetConfig:
    dataset_type: DataSet
    source_dataset_path: str
    noise_dataset_path: str
    output_dataset_path: str
    is_packetloss: bool
    snr_list: List[int] = field(default_factory=lambda: [-10, -6, -3, 0, 3, 6, 10])
    signal_length: int = 16
    snr_segment: int = 1
    predefined_samplerate: int = 16000
    num_examples: int = 100
    num_noises: int = 5


def power_data(data, segment, predefined_samplerate):
    segment_length = int(segment * predefined_samplerate)
    max_in_segment = []
    segments_range = np.arange(segment_length, data.shape[0], segment_length)
    for i, end_segment in enumerate(segments_range):
        max_in_segment.append(np.max(np.abs(data[i * segment_length:end_segment])))
    return np.mean(np.power(max_in_segment, 2))


def find_mul_snr(required_snr_db, signal, noise, cfg):
    snr = 10 ** (required_snr_db / 10)
    return np.sqrt(power_data(signal, cfg.snr_segment, cfg.predefined_samplerate) / (
                snr * power_data(noise, cfg.snr_segment, cfg.predefined_samplerate)))


def calc_audio_length(major_source, noise_path, cfg):
    shape_list = []
    data, samplerate = librosa.load(major_source, sr=cfg.predefined_samplerate)  # dtype='float32'
    shape_list.append(data.shape[0])
    dir_noise = os.listdir(noise_path)
    for i, vid_path in enumerate(dir_noise):
        data, samplerate = librosa.load(f'{noise_path}/{vid_path}', sr=cfg.predefined_samplerate)  # dtype='float32'
        shape_list.append(data.shape[0])
    print(np.min(shape_list) / cfg.predefined_samplerate)
    return int(np.floor(min(np.min(shape_list) / cfg.predefined_samplerate, cfg.signal_length)))


def choose_longest_audio_from_librispech(path_speaker, predefined_samplerate):
    speaker_root_dir = os.listdir(path_speaker)
    speaker_root_dir = ignor_ds_store(speaker_root_dir)
    file_length = 0
    file_path = ""
    for f in speaker_root_dir:
        dir_speaker = os.listdir(f'{path_speaker}/{f}')
        dir_speaker = ignor_ds_store(dir_speaker)
        for d in dir_speaker:
            if d.split(".")[1] != "flac":
                continue
            data, samplerate = librosa.load(f'{path_speaker}/{f}/{d}', sr=predefined_samplerate)
            if file_length < data.shape[0]:
                file_length = data.shape[0]
                file_path = f'{path_speaker}/{f}/{d}'
    return file_path


def ignor_ds_store(dir_path):
    return [dir_path[x] for x in range(len(dir_path)) if dir_path[x].split(".")[-1] != "DS_Store"]


def create_data_snr_list(input_path, output_path, cfg):
    for snr in tqdm(cfg.snr_list):
        print(f'snr: {snr}')
        if not os.path.isdir(f'{output_path}snr_{str(snr)}'):
            os.mkdir(f'{output_path}snr_{str(snr)}')
        input_file_dir = ignor_ds_store(os.listdir(input_path))
        for example in range(len(input_file_dir)):
            if not os.path.isdir(f'{output_path}/snr_{str(snr)}/example_{example}'):
                os.mkdir(f'{output_path}/snr_{str(snr)}/example_{example}')
            create_data_snr(f'{input_path}example_{example}', f'{output_path}snr_{str(snr)}/example_{example}/', snr,
                            cfg)
    return


def create_data_snr(input_path, output_path, snr, cfg):
    noise_path = f'{input_path}/noises'
    signal_num_sec = calc_audio_length(f'{input_path}/source/s.wav', noise_path, cfg)
    data, samplerate = librosa.load(f'{input_path}/source/s.wav', sr=cfg.predefined_samplerate)  # dtype='float32'
    major_source = data[:signal_num_sec * cfg.predefined_samplerate]

    noises = {}

    dir_noises = os.listdir(noise_path)

    normalize_factor = 0
    for i, vid_path in enumerate(dir_noises):
        id = vid_path.split(".")[0]
        data, samplerate = librosa.load(f'{noise_path}/{vid_path}', sr=cfg.predefined_samplerate)  # dtype='float32'
        noise = data[:signal_num_sec * cfg.predefined_samplerate]
        signal = np.zeros_like(major_source, dtype='float32')
        signal += major_source.copy()
        a = find_mul_snr(snr, signal, noise, cfg)
        noises[id] = a
        # print(a)
        signal += a * noise
        if normalize_factor < np.max(np.abs(signal)):
            normalize_factor = np.max(np.abs(signal))
    if not os.path.isdir(f'{output_path}clean_signal/'):
        os.mkdir(f'{output_path}/clean_signal/')
    wavfile.write(f'{output_path}/clean_signal/'
                  f"s.wav", cfg.predefined_samplerate, major_source / normalize_factor)
    for j, vid_path in enumerate(dir_noises):
        id = vid_path.split(".")[0]
        data, samplerate = librosa.load(f'{noise_path}/{vid_path}', sr=cfg.predefined_samplerate)  # dtype='float32'
        noise = data[:signal_num_sec * cfg.predefined_samplerate]
        signal = np.zeros_like(major_source, dtype='float32')
        signal += major_source.copy()
        signal += noises[id] * noise
        signal = signal / normalize_factor
        if not os.path.isdir(f'{output_path}/noise/'):
            os.mkdir(f'{output_path}/noise/')
        wavfile.write(f'{output_path}/noise/'
                      f"{id}.wav", cfg.predefined_samplerate, noise / normalize_factor)
        if not os.path.isdir(f'{output_path}/signal+noise/'):
            os.mkdir(f'{output_path}signal+noise/')
        wavfile.write(f'{output_path}/signal+noise/'
                      f"signal_and_{id}.wav", cfg.predefined_samplerate, signal)


def choose_5_independent_audio_files_from_DEMAND_noises(path_noises, path_dest):
    dir = ignor_ds_store(os.listdir(path_noises))
    noises_kind = random.sample(range(len(dir)), 5)
    for i in noises_kind:
        noises_folder = ignor_ds_store(os.listdir(f'{path_noises}/{dir[i]}'))
        noise = random.randint(0, len(noises_folder) - 1)
        x = f'{path_noises}/{dir[i]}/{noises_folder[noise]}'
        y = f'{path_dest}/{dir[i]}.wav'
        shutil.copyfile(x, y)


def random_not_silent_signals_audioset(path, cfg):
    noises_folder = os.listdir(path)
    noises = np.random.choice(np.arange(0, len(noises_folder)), size=500, replace=False)
    a = np.setdiff1d(np.arange(0, len(noises_folder)), noises)
    for i, n in enumerate(noises):
        data, _ = librosa.load(f'{path}{noises_folder[n]}', sr=cfg.predefined_samplerate)
        while data.max() == 0 and data.min() == 0 and (data.shape[0] / cfg.predefined_samplerate) <= 10:
            new = np.random.choice(a, size=1, replace=False)[0]
            data, _ = librosa.load(f'{path}{noises_folder[new]}', sr=cfg.predefined_samplerate)
            a = np.setdiff1d(a, new)
            noises[i] = new
    return noises


def create_folders_clean_signal_and_noises(data_output_path, example):
    if not os.path.isdir(f'{data_output_path}/{example}'):
        os.mkdir(f'{data_output_path}/{example}')
    if not os.path.isdir(f'{data_output_path}/{example}/signal+noise'):
        os.mkdir(f'{data_output_path}/{example}/signal+noise')
    if not os.path.isdir(f'{data_output_path}/{example}/clean_signal'):
        os.mkdir(f'{data_output_path}/{example}/clean_signal')


def create_examples_folders(data_output_path, example):
    if not os.path.isdir(f'{data_output_path}example_{str(example)}'):
        os.mkdir(f'{data_output_path}example_{str(example)}')
    if not os.path.isdir(f'{data_output_path}example_{str(example)}/noises'):
        os.mkdir(f'{data_output_path}example_{str(example)}/noises')
    if not os.path.isdir(f'{data_output_path}example_{str(example)}/source'):
        os.mkdir(f'{data_output_path}example_{str(example)}/source')


def create_data_folders(path: str, folder: str, dataset_type: str) -> str:
    if not os.path.isdir(f'{path}{folder}'):
        os.mkdir(f'{path}{folder}')
    if not os.path.isdir(f'{path}/{folder}/{dataset_type}'):
        os.mkdir(f'{path}/{folder}/{dataset_type}')
    return f'{path}{folder}/{dataset_type}/'


def remove_directory(directory_path: str):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' has been removed.")
    else:
        print(f"Directory '{directory_path}' does not exist.")


def music_signal_as_source(music_source_path, dir_music, example, cfg, data_split_to_examples_path):
    source_path = f'{music_source_path}{dir_music[example]}'
    try:
        data, _ = librosa.load(source_path, sr=cfg.predefined_samplerate)
    except:
        print(f'Could not load {dir_music[example]} instead took {dir_music[example - 1]}')
        source_path = f'{music_source_path}{dir_music[example - 1]}'
        data, _ = librosa.load(source_path, sr=cfg.predefined_samplerate)
    # the start of the song tend to be silent so i took 30 sec in the song
    if (data.shape[0] / cfg.predefined_samplerate) > 46:
        wavfile.write(f'{data_split_to_examples_path}example_{str(example)}/source/s.wav', cfg.predefined_samplerate,
                      data[30 * cfg.predefined_samplerate:])
    else:
        shutil.copyfile(source_path, f'{data_split_to_examples_path}example_{str(example)}/source/s.wav')


def speech_signal_as_source(source_path, dir_source, example, cfg, data_split_to_examples_path):
    chosen_file_path = choose_longest_audio_from_librispech(f'{source_path}{dir_source[example]}',
                                                            cfg.predefined_samplerate)
    shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/source/s.wav')


def source_music_noise_speech(cfg: CreateDatesetConfig):
    speech_noises_path = f'{cfg.noise_dataset_path}train-clean-360/'
    music_source_path = f'{cfg.source_dataset_path}train/'
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)

    dir_speech = os.listdir(speech_noises_path)
    dir_music = os.listdir(music_source_path)
    for example in tqdm(range(cfg.num_examples)):
        create_examples_folders(data_split_to_examples_path, example)
        for i in range(cfg.num_noises):
            dir_speech = ignor_ds_store(dir_speech)
            chosen_file_path = choose_longest_audio_from_librispech(
                f'{speech_noises_path}/{dir_speech[example * 5 + i]}', cfg.predefined_samplerate)
            shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/noises/{i}.wav')
        source_path = f'{music_source_path}{dir_music[example]}'
        try:
            data, _ = librosa.load(source_path, sr=cfg.predefined_samplerate)
        except:
            print(f'Could not load {dir_music[example]} instead took {dir_music[example - 1]}')
            source_path = f'{music_source_path}{dir_music[example - 1]}'
            data, _ = librosa.load(source_path, sr=cfg.predefined_samplerate)
        # the start of the song tend to be silent so i took 30 sec in the song
        if (data.shape[0] / cfg.predefined_samplerate) > 46:
            wavfile.write(f'{data_split_to_examples_path}example_{str(example)}/source/s.wav',
                          cfg.predefined_samplerate, data[30 * cfg.predefined_samplerate:])
        else:
            shutil.copyfile(source_path, f'{data_split_to_examples_path}example_{str(example)}/source/s.wav')
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    # remove_directory(data_split_to_examples_path)
    return dataset_path


def source_speech_noise_speech(cfg: CreateDatesetConfig):
    speech_noises_path = f"{cfg.noise_dataset_path}train-other-500/"
    speech_source_path = f"{cfg.source_dataset_path}train-clean-360/"
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)
    dir_speech_noise = os.listdir(speech_noises_path)
    dir_speech_source = os.listdir(speech_source_path)
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        for i in range(5):
            chosen_file_path = choose_longest_audio_from_librispech(
                f'{speech_noises_path}{dir_speech_noise[example * 5 + i]}', cfg.predefined_samplerate)
            shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/noises/{i}.wav')
        chosen_file_path = choose_longest_audio_from_librispech(f'{speech_source_path}{dir_speech_source[example]}',
                                                                cfg.predefined_samplerate)
        shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/source/s.wav')
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_speech_noise_DEMAND(cfg: CreateDatesetConfig):
    source_path = f"{cfg.source_dataset_path}train-other-500/"
    noises_path = f"{cfg.noise_dataset_path}16k/"
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)
    dir_source = ignor_ds_store(os.listdir(source_path))
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        noises_dest_path = f'{data_split_to_examples_path}example_{str(example)}/noises'
        choose_5_independent_audio_files_from_DEMAND_noises(noises_path, noises_dest_path)
        # add speech
        chosen_file_path = choose_longest_audio_from_librispech(f'{source_path}{dir_source[example]}',
                                                                cfg.predefined_samplerate)
        shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/source/s.wav')
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_music_noise_DEMAND(cfg: CreateDatesetConfig):
    noises_path = f"{cfg.noise_dataset_path}16k/"
    source_path = f'{cfg.source_dataset_path}train/'
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)
    dir_source = ignor_ds_store(os.listdir(source_path))
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        noises_dest_path = f'{data_split_to_examples_path}example_{str(example)}/noises'
        choose_5_independent_audio_files_from_DEMAND_noises(noises_path, noises_dest_path)
        # add music
        music_signal_as_source(source_path, dir_source, example, cfg, data_split_to_examples_path)
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_speech_noise_AudioSet(cfg: CreateDatesetConfig):
    noises_path = f"{cfg.noise_dataset_path}"
    source_path = f"{cfg.source_dataset_path}train-clean-360/"
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)
    dir_source = ignor_ds_store(os.listdir(source_path))
    noises_folder = ignor_ds_store(os.listdir(noises_path))
    noises = random_not_silent_signals_audioset(noises_path, cfg)
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        noises_dest_path = f'{data_split_to_examples_path}example_{str(example)}/noises'
        for i in range(5):
            x = f'{noises_path}{noises_folder[noises[example * 5 + i]]}'
            y = f'{noises_dest_path}/{noises_folder[noises[example * 5 + i]]}'
            shutil.copyfile(x, y)
        # add speech
        speech_signal_as_source(source_path, dir_source, example, cfg, data_split_to_examples_path)
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_music_noise_AudioSet(cfg):
    noises_path = f"{cfg.noise_dataset_path}"
    source_path = f'{cfg.source_dataset_path}train/'
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      cfg.dataset_type.value)

    dir_source = ignor_ds_store(os.listdir(source_path))
    dir_noises = ignor_ds_store(os.listdir(noises_path))
    noises = random_not_silent_signals_audioset(noises_path, cfg)
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        noises_dest_path = f'{data_split_to_examples_path}example_{str(example)}/noises'
        for i in range(5):
            x = f'{noises_path}{dir_noises[noises[example * 5 + i]]}'
            y = f'{noises_dest_path}/{dir_noises[noises[example * 5 + i]]}'
            shutil.copyfile(x, y)
            shutil.copyfile(x, y)
        # add music
        music_signal_as_source(source_path, dir_source, example, cfg, data_split_to_examples_path)

    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_music_noise_speech_multi_microphone(cfg):
    noises_path = f'{cfg.noise_dataset_path}train-other-500//'
    source_path = f'{cfg.source_dataset_path}train/'
    data_split_to_examples_path = create_data_folders(cfg.output_dataset_path, "data_split_to_examples",
                                                      f'{cfg.dataset_type.value}_multi_microphone')
    dir_noise = ignor_ds_store(os.listdir(noises_path))
    dir_source = ignor_ds_store(os.listdir(source_path))
    for example in tqdm(range(100)):
        create_examples_folders(data_split_to_examples_path, example)
        for i in range(10):
            chosen_file_path = choose_longest_audio_from_librispech(f'{noises_path}{dir_noise[example * 10 + i]}',
                                                                    cfg.predefined_samplerate)
            shutil.copyfile(chosen_file_path, f'{data_split_to_examples_path}example_{str(example)}/noises/{i}.wav')
        # add music
        music_signal_as_source(source_path, dir_source, example, cfg, data_split_to_examples_path)
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", cfg.dataset_type.value)
    create_data_snr_list(data_split_to_examples_path, dataset_path, cfg)
    return dataset_path


def source_vs_noise_paketloss(input_path, cfg):
    dataset_path = create_data_folders(cfg.output_dataset_path, "snr_dataset", f'{cfg.dataset_type.value}_packetloos')
    # run on SNR folders
    dir_snr = ignor_ds_store(os.listdir(input_path))
    for snr in tqdm(dir_snr):
        if not os.path.isdir(f'{dataset_path}{snr}'):
            os.mkdir(f'{dataset_path}{snr}')
        dir_examples = ignor_ds_store(os.listdir(f'{input_path}{snr}'))
        for example in dir_examples:
            create_folders_clean_signal_and_noises(f'{dataset_path}{snr}', example)
            clean_signal, samplerate = sf.read(f'{input_path}{snr}/{example}/clean_signal/s.wav')
            clean_signal = clean_signal[:8 * samplerate]
            wavfile.write(f'{dataset_path}{snr}/{example}/clean_signal/s.wav', cfg.predefined_samplerate, clean_signal)
            dir_signal_noise = os.listdir(f'{input_path}{snr}/{example}/signal+noise/')
            random_segments = np.random.choice(np.arange(0, 8), size=5, replace=False)
            for i, signal_noise in enumerate(dir_signal_noise):
                data, samplerate = sf.read(f'{input_path}{snr}/{example}/signal+noise/{signal_noise}')
                signal = data[:8 * samplerate]
                segment_values = 0.001 * np.ones((samplerate,)) * np.random.randn(samplerate)
                signal[random_segments[i] * samplerate:(random_segments[i] + 1) * samplerate] = segment_values
                wavfile.write(f'{dataset_path}{snr}/{example}/signal+noise/{signal_noise}', cfg.predefined_samplerate,
                              signal)
    return dataset_path


def detasets_factory(cfg: CreateDatesetConfig):
    dataset_path = cfg.output_dataset_path
    if cfg.dataset_type == DataSet.SOURCE_MUSIC_NOISE_SPEECH:
        dataset_path = source_music_noise_speech(cfg)
    if cfg.dataset_type == DataSet.SOURCE_MUSIC_NOISE_AUDIOSET:
        dataset_path = source_music_noise_AudioSet(cfg)
    if cfg.dataset_type == DataSet.SOURCE_MUSIC_NOISE_DEMAND:
        dataset_path = source_music_noise_DEMAND(cfg)
    if cfg.dataset_type == DataSet.SOURCE_SPEECH_NOISE_SPEECH:
        dataset_path = source_speech_noise_speech(cfg)
    if cfg.dataset_type == DataSet.SOURCE_SPEECH_NOISE_AUDIOSET:
        dataset_path = source_speech_noise_AudioSet(cfg)
    if cfg.dataset_type == DataSet.SOURCE_SPEECH_NOISE_DEMAND:
        dataset_path = source_speech_noise_DEMAND(cfg)
    return dataset_path


@pyrallis.wrap()
def main(cfg: CreateDatesetConfig):
    if cfg.is_packetloss:
        if os.path.exists(f"{cfg.output_dataset_path}snr_dataset/{cfg.dataset_type.value}"):
            dataset_path = source_vs_noise_paketloss(f"{cfg.output_dataset_path}snr_dataset/{cfg.dataset_type.value}/",
                                                     cfg)
        else:
            dataset_path = detasets_factory(cfg)
            dataset_path = source_vs_noise_paketloss(dataset_path, cfg)
    elif cfg.dataset_type == DataSet.SOURCE_MUSIC_NOISE_SPEECH_CHANGED_NUMBER_OF_NOISES:
        dataset_path = source_music_noise_speech_multi_microphone(cfg)
    else:
        dataset_path = detasets_factory(cfg)
    print(f'the final path to the dataset: {dataset_path}')
    return


if __name__ == '__main__':
    main()
