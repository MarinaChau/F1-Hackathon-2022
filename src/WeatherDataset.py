import numpy as np
import tensorflow as tf


class WeatherDataset:

    def __init__(self, sequence_length, df, batch_size, instanciate_data=True):
        """Constructor for WeatherDataset class.

        Args:
            sequence_length (int): number of timesteps used for one example
            df (pd.DataFrame): input data
            batch_size (int): batch size for the tf.data.Dataset
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.offsets = np.array([5, 10, 15, 30, 60])*60
        if instanciate_data:
          self.x_array, self.y_categories_array, self.y_probabilities_array = self.process_inputs(
              df)
        
    def import_data(self, x, y1, y2):
      self.x_array = x
      self.n_features = x[0].shape[0]
      self.y_categories_array = y1
      self.y_probabilities_array = y2

    def process_inputs(self, df):
        """Generates input array as well as two label arrays from an input DataFrame.

        Args:
            df (pd.DataFrame): input data

        Returns:
            X (np.ndarray): input data as array
            label1 (np.ndarray): one_hot weather category label
            label2 (np.ndarray): rain probability label
        """
        X, label1, label2 = [], [], []
        df = df.iloc[::-1].copy()
        df["CURRENT_TIME"] = df["M_SESSION_DURATION"] - df["M_SESSION_TIME_LEFT"]
        current_time = df["CURRENT_TIME"].to_numpy().astype(np.int64)
        
        
        X = list(df.drop(columns=["M_SESSION_DURATION", "M_SESSION_TIME_LEFT"]).values)
        self.n_features = X[0].shape[0]

        for i in range(df.shape[0]-1):
            if i%1000 == 0:
              print(f"{i}/{df.shape[0]}")
            # Array of features, labels are created later.
            offset_label1, offset_label2 = [], []
            for offset in self.offsets:
                # print(f"Processing for offset {offset}...")
                # print(f"Current time : {current_time[i]}")
                # print(df["CURRENT_TIME"].iloc[i])
                sess_id = df["M_SESSION_LINK_IDENTIFIER"].iloc[i]
                # print(sess_id)
                candidates = df[df["M_SESSION_LINK_IDENTIFIER"] == sess_id]["CURRENT_TIME"] 
                # print(candidates)
                # print(candidates - (df["CURRENT_TIME"].iloc[i] + offset))
                indx = np.argmin(np.abs(candidates - (df["CURRENT_TIME"].iloc[i] + offset)))
                # session["M_SESSION_DURATION"] - session["M_SESSION_TIME_LEFT"]
                # indx = np.argmin(df[df["M_SESSION_LINK_IDENTIFIER"] == sess_id]["CURRENT_TIME"] - (df["CURRENT_TIME"].iloc[i] + offset))
                # print(indx)
                # indx = np.abs(
                #     current_time - (current_time[i] + offset)).argmin()
                # print(f"Index of GT: {indx}")
                # print(df.iloc[indx])
                m_weather = df["M_WEATHER"].iloc[indx]
                offset_onehot = np.zeros(6)
                offset_onehot[m_weather] = 1
                # print(f"GT m_weather : {np.argmax(offset_onehot)}")

                # if m_weather != 5:  
                #     offset_onehot[m_weather] = 1
                # else:  # There are only 0, 1, 2, 5
                #     offset_onehot[3] = 1
                # offset_rain_percentage = np.zeros(1)
                offset_rain_percentage = df['M_RAIN_PERCENTAGE'].iloc[indx]/100
                offset_label1.append(offset_onehot)
                offset_label2.append(offset_rain_percentage)
            label1.append(offset_label1)
            label2.append(offset_label2)
        
        return np.array(X), np.array(label1), np.array(label2)[:, :, np.newaxis]

    def create_dataset(self):
        """
        Generates dataset.
        Uses class attributes to create self.data which is a tf.data.Dataset.
        """
        self.x = self.build_data(
            self.x_array, sequence_length=self.sequence_length)
        self.y_categories = [self.build_data(
            np.roll(self.y_categories_array[:,i], self.sequence_length, axis=0), sequence_length=1) for i in range(5)]
        self.y_categories = [e.map(
            lambda x: tf.reshape(tf.squeeze(x), [self.batch_size, 6])) for e in self.y_categories]
        self.y_probabilities = self.build_data(
            np.roll(self.y_probabilities_array, self.sequence_length, axis=0), sequence_length=1)
        self.y_probabilities = self.y_probabilities.map(
            lambda x: tf.squeeze(x))
        self.data = tf.data.Dataset.zip(
            (self.x, (*self.y_categories, self.y_probabilities)))

    def build_data(self, array, sequence_length, stride=1, shuffle=False):
        return tf.keras.utils.timeseries_dataset_from_array(data=array,
                                                            targets=None,
                                                            sequence_length=sequence_length,
                                                            sequence_stride=int(self.sequence_length/2),
                                                            shuffle=shuffle,
                                                            batch_size=self.batch_size,
                                                            end_index=array.shape[0]-8*self.sequence_length-1)
