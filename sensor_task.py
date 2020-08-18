import numpy as np 
import tensorflow as tf 
import sys 
import pickle 

import pandas as pd 



class Sensor():

    def __init__(self, sensor_type, training_percentage=0.95, selected_hour = 'all', seed=0):
        # sensor_type in {'temp', 'humidity', 'light', 'voltage'}
        self.xdim = 2
        self.ydim = 1
        self.xmin = 0.0
        self.xmax = 1.0

        self.sensor_type = sensor_type

        self.df = pd.read_csv("sensor_data/sufficient_sensor_data.csv", header=0)

        if sensor_type == "temp":
            self.df = self.df[self.df.temp < 50]
            self.df = self.df[self.df.temp >= 10]
        elif sensor_type == "humidity":
            self.df = self.df[self.df.humidity < 60]
            self.df = self.df[self.df.humidity > 10]
        elif sensor_type == "voltage":
            self.df = self.df[self.df.voltage < 3]
            self.df = self.df[self.df.voltage > 2]
        elif sensor_type == "light":
            self.df = self.df[self.df.light <= 100000]
            self.df = self.df[self.df.light >= 0]

        self.df['moteid'] = (self.df['moteid'] - 1).astype(int)

        self.loc = pd.read_csv("sensor_data/mote_locs.txt", delimiter=" ", header=None)
        self.loc.columns = ["moteid", "x", "y"]
        self.loc['moteid'] = (self.loc['moteid'] - 1).astype(int)

        # remove those moteid not in self.loc
        self.df = self.df[ self.df.moteid <= self.loc['moteid'].max() ]
        self.df = self.df[ self.df.moteid >= self.loc['moteid'].min() ]
        
        self.loc.set_index('moteid', inplace=True)

        # normalize X, Y to range [0,1]
        # so xmin = 0, xmax = 1
        max_x = self.loc['x'].max()
        min_x = self.loc['x'].min()
        max_y = self.loc['y'].max()
        min_y = self.loc['y'].min()

        self.loc['x'] = (self.loc['x'] - min_x) / (max_x - min_x)
        self.loc['y'] = (self.loc['y'] - min_y) / (max_y - min_y)

        self.moteid_to_loc = np.zeros([self.df['moteid'].max()+1, 2])
        for i in range(self.df['moteid'].max()+1):
            self.moteid_to_loc[i,:] = self.loc.loc[i].to_numpy()


        if sensor_type == "light":
            print("using log of {} measurement".format(sensor_type))
            self.df[sensor_type] = np.log(self.df[sensor_type] + 0.1)

            if selected_hour != 'all':
                print("restrict measurement to 1 hour of the day: {}".format(selected_hour))
                self.df = self.df[ self.df['hour'] == selected_hour ]


        # remove all date_hour_minute has < 35 unique moteids
        nunique_moteids = self.df.groupby('date_hour_minute').moteid.nunique()
        nunique_moteids = nunique_moteids[ nunique_moteids > 35 ]
        self.keys = np.unique(nunique_moteids.index.values)

        print("Separating training and test task set with random seed: {}".format(seed))
        np.random.seed(seed)
        np.random.shuffle(self.keys)
        n_training_keys = int(len(self.keys) * training_percentage)
        self.training_keys = self.keys[:n_training_keys]
        self.test_keys = self.keys[n_training_keys:]

        print("Training percentage: {}".format(training_percentage))
        print("Training task set size: {}".format(len(self.training_keys)))
        print("Test task set size: {}".format(len(self.test_keys)))

        if selected_hour != "all":
            save_data_by_dhm_filename = "data_by_dhm_{}_hour{}.p".format(sensor_type, selected_hour)
        else:
            save_data_by_dhm_filename = "data_by_dhm_{}.p".format(sensor_type)

        try:
            with open(save_data_by_dhm_filename, 'rb') as infile:
                self.data_by_dhm = pickle.load(infile)

            print("load data_by_dhm from {}".format(save_data_by_dhm_filename))
            sys.stdout.flush()

        except:

            print("creating file {}".format(save_data_by_dhm_filename))
            sys.stdout.flush()

            df_date_hour_minute = self.df.groupby('date_hour_minute')
            self.data_by_dhm = {}
            for key in self.keys:
                df = df_date_hour_minute.get_group(key)
                df_mean = df.groupby('moteid').mean()

                data_for_key = np.zeros([df_mean.shape[0], 3])
                # x0, x1, y

                count = 0
                for moteid,row in df_mean.iterrows():
                    y = row[sensor_type]
                    x = self.moteid_to_loc[moteid,:]
                    data_for_key[count,:2] = x 
                    data_for_key[count,2] = y
                    count += 1

                self.data_by_dhm[key] = data_for_key

            with open(save_data_by_dhm_filename, "wb") as outfile:
                pickle.dump(self.data_by_dhm, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done init"); sys.stdout.flush()


    def get_random_datasets(self, ndataset, dataset_size, is_training=True):

        if dataset_size > 35:
            print("Not all timestamp has more than 35 data points, cannot obtain all dataset_size = {}".format(dataset_size))
            dataset_size = 35

        keys = self.training_keys if is_training else self.test_keys

        if ndataset > len(keys):
            print("Require {} tasks more than {} available {} tasks".format(
                ndataset,
                len(keys),
                "training" if is_training else "test"
            ))

            ndataset = len(keys)


        # shuffle the datasets
        selected_keys = keys[np.random.randint(len(keys), size=ndataset)]

        X = []
        Y = []

        ground_truths = {}

        for i in range(ndataset):

            key = selected_keys[i]
            data_for_key = self.data_by_dhm[key]

            ndata = data_for_key.shape[0]
            idxs = np.array(list(range(ndata)))

            # shuffle data within a dataset
            np.random.shuffle(idxs)
            
            selected_idxs = idxs[:dataset_size]

            X.append( data_for_key[selected_idxs,:2] )
            Y.append( data_for_key[selected_idxs,2:] )

            ground_truths[key] = data_for_key

        X = np.stack(X)
        # (ndataset, dataset_size, xdim)
        Y = np.stack(Y)
        # (ndataset, dataset_size, ydim)

        return X, Y, ground_truths


    def get_y_distance(self, y0, y1):
        # y0: (...,ntask0, ntask1|1, npoint, ydim)
        # y1: (...,ntask0|1, ntask1, npoint, ydim)
        return tf.reduce_mean(
                        (y0 - y1) * (y0 - y1),
                        axis = -1)
        # (...,ntask0, ntask1, npoint)


    def get_random_datasets_by_keys(self, selected_keys, selected_idxs, dataset_size):

        if dataset_size > 35:
            print("Not all timestamp has more than 35 data points, cannot obtain all dataset_size = {}".format(dataset_size))
            dataset_size = 35
        # assert dataset_size <= 35

        ndataset = len(selected_keys)

        X = []
        Y = []

        ground_truths = {}

        for i in range(ndataset):

            key = selected_keys[i]
            data_for_key = self.data_by_dhm[key]
            
            idxs = selected_idxs[i]

            X.append( data_for_key[idxs,:2] )
            Y.append( data_for_key[idxs,2:] )

            ground_truths[key] = data_for_key

        X = np.stack(X)
        # (ndataset, dataset_size, xdim)
        Y = np.stack(Y)
        # (ndataset, dataset_size, ydim)

        return X, Y, ground_truths
