import os, json, mne
import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.signal import resample


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def interpolate_blinks(gaze, times, labels, sr):
    """
    Function to linearly interpolate blinks
    """
    start, end, flag = None, None, False
    for i, label in enumerate(labels):
        if label == "saccadeStart":
            start = np.int32(times[i])
        elif label == "saccadeEnd":
            lag = int(0.1 * sr)
            start = max(0, start - lag)
            end = min(np.int32(times[i]) + lag, len(gaze) - 1)
            if flag:
                # interpolate linearly
                gaze[start:end] = np.linspace(gaze[start], gaze[end], end - start)
                flag = False
        elif label == "blinkStart":
            flag = True
    return gaze


def load_markers(
    annot: mne.Annotations, mapping: dict, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load markers from the data annotations and map them to labels.
    Mapping is done using the provided dictionary (from EEG study).
    Markers include eye tracking events and stimuli markers.
    Output is sorted by time and disentangled for identical times.

    Parameters
    ----------
    annot : mne.Annotations
    mapping : dict
        Mapping of stimuli markers to labels.
    sr : int
        Sampling rate of the data.

    Returns
    -------
    tuple of np.ndarray
        Sorted times and labels of the markers.
    """
    # extract, combine, and sort keys
    annot_indices = []
    for number in mapping.keys():
        this_list = np.where(annot.description == f"S{number}")[0]
        annot_indices += list(this_list)
    annot_indices = np.array(annot_indices)

    # extract eye tracking markers
    saccadeStart = np.where(annot.description == "saccade")[0]
    saccadeEnd = [(annot.onset[i] + annot.duration[i]) * sr for i in saccadeStart]
    saccadeStart = np.array([annot.onset[i] * sr for i in saccadeStart])
    saccadeEnd = np.array(saccadeEnd)

    fixationStart = np.where(annot.description == "fixation")[0]
    fixationEnd = [(annot.onset[i] + annot.duration[i]) * sr for i in fixationStart]
    fixationStart = np.array([annot.onset[i] * sr + 1 for i in fixationStart])
    fixationEnd = np.array(fixationEnd)

    blinkStart = np.where(annot.description == "BAD_blink")[0]
    blinkEnd = [(annot.onset[i] + annot.duration[i]) * sr for i in blinkStart]
    blinkStart = np.array([annot.onset[i] * sr for i in blinkStart])
    blinkEnd = np.array(blinkEnd)

    # start and end of recording
    startRec = np.where(annot.description == "Recording Start")[0]
    endRec = np.where(annot.description == "Stop recording")[0]

    # extract times in samples
    times = annot.onset[annot_indices] * sr
    times = np.concatenate([times, saccadeStart])
    times = np.concatenate([times, saccadeEnd])
    times = np.concatenate([times, fixationStart])
    times = np.concatenate([times, fixationEnd])
    times = np.concatenate([times, blinkStart])
    times = np.concatenate([times, blinkEnd])
    times = np.concatenate([times, annot.onset[startRec] * sr])
    times = np.concatenate([times, annot.onset[endRec] * sr])
    times = times.astype(np.uint32)

    # extract text labels
    labels = annot.description[annot_indices]
    labels = [mapping[l[1:]] for l in labels]
    labels = np.concatenate([labels, ["saccadeStart"] * len(saccadeStart)])
    labels = np.concatenate([labels, ["saccadeEnd"] * len(saccadeEnd)])
    labels = np.concatenate([labels, ["fixationStart"] * len(fixationStart)])
    labels = np.concatenate([labels, ["fixationEnd"] * len(fixationEnd)])
    labels = np.concatenate([labels, ["blinkStart"] * len(blinkStart)])
    labels = np.concatenate([labels, ["blinkEnd"] * len(blinkEnd)])
    labels = np.concatenate([labels, ["startRec"] * len(startRec)])
    labels = np.concatenate([labels, ["endRec"] * len(endRec)])

    # sort by time
    times, labels = zip(*sorted(zip(times, labels)))
    times, labels = np.array(times), np.array(labels)

    # disentagle identical times
    for i, t in enumerate(times):
        if i and t == times[i - 1]:
            if "Start" in labels[i]:
                times[i] += 1

    # sort again
    times, labels = zip(*sorted(zip(times, labels)))
    times, labels = np.array(times), np.array(labels)
    return times, labels


def get_markers_st(subject: str, cat: str = "SentimentofSentence") -> np.ndarray:
    # load csv for this subject
    csv_path = "/PATH/TO/eyelink-processed/sent_annotations/"
    sub_file = pd.read_csv(csv_path + f"sub{subject}_preprocessed.csv")

    # get the category of interest
    markers = sub_file[cat].values
    positions = sub_file["Position"].values

    if "Sentiment" in cat:
        markers, positions = [], []
        for i, tr in sub_file.iterrows():
            if tr["SentimentofSentence"] == "negative":
                markers.append("negative")
                positions.append(tr["Position"])
            elif tr["SentimentofWord"] != "negative":
                markers.append("positive")
                positions.append(tr["Position"])
        markers = np.array(markers)
        positions = np.array(positions)

    elif "Congruency" in cat:
        markers = np.where(markers == "Congruent", "positive", "negative")
    else:
        raise ValueError(f"Invalid category {cat} for ST task.")

    return np.array(markers), np.array(positions)


def cache_data(
    asc,
    subject: str,
    mapping: dict,
    path: str,
    timing: str,
    win: tuple[float, float] = (-1.2, 0.6),  # (-0.5, 0.9), (-1.2, 0)
) -> None:
    """
    Cache the data for a subject from the source directory.
    """
    # read MNE files
    fname = os.path.join(asc, f"sub{subject}", f"EY{subject}_{task}.asc")
    eye_data = mne.io.read_raw_eyelink(fname, verbose=False)

    # extract all trials
    signals = eye_data.get_data()[:3].T
    times, labels = load_markers(
        annot=eye_data.annotations, mapping=mapping, sr=sr[subject]
    )
    # interpolate blinks
    signals = interpolate_blinks(signals, times, labels, sr[subject])

    # add the trial markers
    label_pos = np.where(labels == "startOfSentenceMarker")[0]
    times_pos = times[label_pos]

    # load the indices
    with open("/PATH/TO/eyelink-processed/metadata/target_indices_1_7.json", "r") as f:
        target_indices = json.load(f)

    times_mark = times_pos + target_indices[subject]
    mis_mark = "TargetWord_Sentence_marker"
    labels_mark = np.array([mis_mark] * len(times_mark))

    times = np.concatenate([times, times_mark])
    labels = np.concatenate([labels, labels_mark])

    # sort by time
    times, labels = zip(*sorted(zip(times, labels)))
    times, labels = np.array(times), np.array(labels)

    # cap at screen edges
    signals[..., 0] = np.clip(signals[..., 0], 0, 1920)
    signals[..., 1] = np.clip(signals[..., 1], 0, 1080)

    # select trials for each marker
    trials = []
    for i, label in enumerate(labels):
        found_condition = (
            label.startswith("Subject")
            if "resp" in timing
            else "TargetWord_Sentence_marker" in label
        )
        if found_condition:
            start = int(times[i]) + int(win[0] * sr[subject])
            end = int(times[i]) + int(win[1] * sr[subject])
            trials.append(signals[start:end])

    if len(trials) == 320:
        trials = np.stack(trials)
        if task == "EM":
            raise NotImplementedError("EM task not implemented.")
        else:
            markers, positions = get_markers_st(subject)
            trials = trials[positions]
    else:
        print(f"{len(trials)} trials found for {subject} in {task} task.")
        subjects.remove(subject)
        return

    # save data in preprocessed form
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, f"{subject}_{task}_tracks.npy"), trials)
    np.save(os.path.join(path, f"{subject}_{task}_markers.npy"), markers)


if __name__ == "__main__":

    # core variables
    trials_in_set, n_bootstrap, timing, task = 30, 200, "resp", "ST"
    seq_len = int(1.4 * 250) if timing == "read" else int(1.2 * 250)

    # core paths and task
    name = ""
    main_path = "/PATH/TO/eyelink-processed/"
    asc_path = main_path + "asc_files/"
    stim_path = main_path + "metadata/mapping.json"
    proc_path = main_path + f"st_trials_{timing}/"
    out_path = main_path + f"input_{trials_in_set}trials_{timing}{name}/"
    os.makedirs(proc_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # available subjects and their sampling rates
    subjects = ["001", "002", "003", "004", "005", "006", "007"]
    sr = {s: 500 for s in subjects}

    subjects_copy = subjects.copy()
    trial_dict = {}
    for subject in tqdm(subjects_copy):
        subject_path = os.path.join(proc_path, f"sub{subject}")
        # cache_data(
        #    asc=asc_path,
        #    subject=subject,
        #    mapping=load_json(stim_path),
        #    path=subject_path,
        #    timing=timing,
        # )

        # load subject data and markers
        sub_data = np.load(os.path.join(subject_path, f"{subject}_{task}_tracks.npy"))
        markers = np.load(os.path.join(subject_path, f"{subject}_{task}_markers.npy"))

        # keep only the first % of the data
        # threshold = int(0.75 * len(sub_data))
        # sub_data, markers = sub_data[:threshold], markers[:threshold]

        # downsample to 250 Hz
        sub_data = resample(sub_data, sub_data.shape[1] // 2, axis=1)

        # build the categories according to the task
        categories = None
        if task == "EM":
            raise ValueError("EM task not implemented.")
        elif task == "ST":
            categories = {"positive": [], "negative": []}
            for m in categories.keys():
                m_data = sub_data[np.where(markers == m)[0]]
                categories[m] = np.nan_to_num(m_data)
        else:
            raise ValueError(f"Invalid task {task}.")

        # for each category, prepare 200 aggregate samples
        cat_samples = {
            m: np.zeros((n_bootstrap, trials_in_set * 2, seq_len))
            for m in categories.keys()
        }

        trial_dict[subject] = {}
        for m in categories.keys():
            trial_dict[subject][m] = {}
            this_path = os.path.join(out_path, f"sub{subject}_{m[:3]}.npy")
            for i in range(n_bootstrap):
                rand_idx = np.random.choice(len(categories[m]), trials_in_set)
                trial_dict[subject][m][str(i + 1)] = [int(idx) for idx in rand_idx]

                this_sample = categories[m][rand_idx][..., :2]
                this_sample = np.swapaxes(this_sample, 1, 2)
                this_sample = np.concatenate(this_sample, axis=0)
                cat_samples[m][i] = this_sample

            # save the data as a numpy array
            print(f"Saving {this_path}", cat_samples[m].shape)
            np.save(this_path, cat_samples[m])

    # save the trial dictionary
    dpath = f"/PATH/TO/eyelink-processed/metadata/{task}_{trials_in_set}_{timing}_17{name}.json"
    with open(dpath, "w") as f:
        json.dump(trial_dict, f, indent=4)
