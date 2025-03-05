import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('../PEMS_data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

        # day and week
        day_data = np.zeros_like(data)
        week_data = np.zeros_like(data)
        day_init = 0
        week_init = 0
        for index in range(data.shape[0]):
            if (index) % (288 * 7) == 0:
                week_init = 0
            if (index) % 288 == 0:
                day_init = 0
            if (index) % 288 == 0:
                week_init = week_init + 1
            day_init = day_init + 1
            day_data[index:index + 1, :] = day_init
            week_data[index:index + 1, :] = week_init


    elif dataset == 'PEMS08':
        data_path = os.path.join('../PEMS_data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

        # day and week
        day_data = np.zeros_like(data)
        week_data = np.zeros_like(data)
        day_init = 0
        week_init = 0
        week_spe = 5
        for index in range(data.shape[0]):
            # day add
            if (index) % 288 == 0:
                day_init = 0
            day_init = day_init + 1
            day_data[index:index + 1, :] = day_init
            # week add
            if index < (288*3):
                if index !=0 and index % 288 == 0:
                    week_spe = week_spe + 1
                week_data[index:index + 1, :] = week_spe
            else:
                if (index-288*3) % (288 * 7) == 0:
                    week_init = 0
                if index % 288 == 0:
                    week_init = week_init + 1
                week_data[index:index + 1, :] = week_init

    elif dataset == 'PEMS03':
        data_path = os.path.join('../PEMS_data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

        # day and week
        day_data = np.zeros_like(data)
        week_data = np.zeros_like(data)
        day_init = 0
        week_init = 0
        week_spe = 6
        for index in range(data.shape[0]):
            # day add
            if (index) % 288 == 0:
                day_init = 0
            day_init = day_init + 1
            day_data[index:index + 1, :] = day_init
            # week add
            if index < (288*3):
                if index !=0 and index % 288 == 0:
                    week_spe = week_spe + 1
                week_data[index:index + 1, :] = week_spe
            else:
                if (index-288*3) % (288 * 7) == 0:
                    week_init = 0
                if index % 288 == 0:
                    week_init = week_init + 1
                week_data[index:index + 1, :] = week_init

    elif dataset == 'PEMS07':
        data_path = os.path.join('../PEMS_data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data

        day_data = np.zeros_like(data)
        week_data = np.zeros_like(data)
        day_init = 0
        week_init = 0
        for index in range(data.shape[0]):
            if (index) % (288 * 7) == 0:
                week_init = 0
            if (index) % 288 == 0:
                day_init = 0
            if (index) % 288 == 0:
                week_init = week_init + 1
            day_init = day_init + 1
            day_data[index:index + 1, :] = day_init
            week_data[index:index + 1, :] = week_init

    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1)
        week_data = np.expand_dims(week_data, axis=-1)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data[..., 0:1].shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]))
    return data
