import datetime as datetime
import numpy as np
import pandas as pd
import requests
import json


def query(host, start, metric, end=None, cached_filename='cached.json', **kwargs):
    payload = {'m': metric}
    payload.update(kwargs)

    if type(start) is datetime.date or type(start) is datetime.datetime:
        start = start.strftime("%Y/%m/%d-%H:%M:%S")
    payload['start'] = start

    if type(end) is datetime.date or type(start) is datetime.datetime:
        payload['end'] = end.strftime("%Y/%m/%d-%H:%M:%S")
    elif end:
        payload['end'] = end

    request = requests.get(host, params=payload)
    print('query: ', request.url)

    # store cached version
    with open(cached_filename, 'w') as outfile:
        json.dump(request.json(), outfile)

    return convert(request.json())


def load_cached_json(filename='cached.json'):
    with open(filename) as data_file:
        json_obj = json.load(data_file)
    return convert(json_obj)


def convert(json_obj):
    df = pd.DataFrame()
    for metric in json_obj:
        name = metric['metric']
        datapoints = metric['dps']
        ts = pd.Series(index=[datetime.datetime.fromtimestamp(int(ts)) for ts in datapoints.keys()],
                       data=[float(v) for v in datapoints.values()])
        df[name] = ts
    return df.sort_index()


# noinspection PyTypeChecker
def mask_print_session(fila_distance, smoothing_window=40, distance_thresh=30):

    # smooth and threshold distance values
    fila_distance = fila_distance.rolling(window=smoothing_window).min()
    fila_distance[fila_distance < distance_thresh] = 0

    # detect changes in print
    fila_run = (fila_distance > 10) * 500
    fila_run = np.gradient(fila_run)

    # mask printing sessions
    print_mask = np.nan * np.ones(shape=len(fila_distance))
    print_count = 0
    print_active = False
    for index, fr in enumerate(fila_run):
        if fr > 0:
            if not print_active:
                print_count += 1
            print_active = True
        elif fr < 0:
            print_active = False

        if print_active:
            print_mask[index] = print_count

    return pd.Series(index=fila_distance.index, data=print_mask, dtype=np.int)
