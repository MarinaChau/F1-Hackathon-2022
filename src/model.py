from ctypes import ArgumentError
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

class WeatherModel(Model):

  def __init__(self, sequence_length, n_features, params, model_type='lstm'):
    super().__init__()
    self.sequence_length = sequence_length
    self.n_features = n_features
    self.params = params
    if model_type == 'lstm':
      self.model = self.__build_lstm_model()
    elif model_type == 'gru':
      self.model = self.__build_gru_model()
    else:
      raise ArgumentError("model type must be 'gru' or 'lstm'")

  def __build_lstm_model(self):
    # Dummy values for seq_length and n_columns
    inp = layers.Input((self.sequence_length, self.n_features))
    x = layers.LSTM(self.params["recurrent_cell_shape"], return_sequences=True)(inp)
    x = layers.LSTM(self.params["recurrent_cell_shape"], return_sequences=False)(x)
    x = layers.Dense(256)(x)
    weather_type = [layers.Dense(6, activation="softmax", name=f"weather_type_softmax_{i}")(x) for i in range(5)]
    rain_probabilities = layers.Dense(5, activation='sigmoid', name='rain_prob')(x)
    lstm_model = Model(inp, [*weather_type, rain_probabilities])
    return lstm_model

  def __build_gru_model(self):
    # Dummy values for seq_length and n_columns
    inp = layers.Input((self.sequence_length, self.n_features))
    x = layers.GRU(self.params["recurrent_cell_shape"], return_sequences=True)(inp)
    x = layers.GRU(self.params["recurrent_cell_shape"], return_sequences=False)(x)
    x = layers.Dense(256)(x)
    weather_type = [layers.Dense(6, activation="softmax", name=f"weather_type_softmax_{i}")(x) for i in range(5)]
    rain_probabilities = layers.Dense(5, activation='sigmoid', name='rain_prob')(x)
    gru_model = Model(inp, [*weather_type, rain_probabilities])
    return gru_model

  def __build_metrics(self):
    metrics = {}
    metrics["rain_prob"] = tf.keras.metrics.MeanAbsoluteError()
    metrics["weather_type_softmax_0"] = tf.keras.metrics.CategoricalAccuracy()
    metrics["weather_type_softmax_1"] = tf.keras.metrics.CategoricalAccuracy()
    metrics["weather_type_softmax_2"] = tf.keras.metrics.CategoricalAccuracy()
    metrics["weather_type_softmax_3"] = tf.keras.metrics.CategoricalAccuracy()
    metrics["weather_type_softmax_4"] = tf.keras.metrics.CategoricalAccuracy()
    return metrics

  def compile(self):
    self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.params["learning_rate"]),
                       loss = {"weather_type_softmax_0":"categorical_crossentropy",
                               "weather_type_softmax_1":"categorical_crossentropy",
                               "weather_type_softmax_2":"categorical_crossentropy",
                               "weather_type_softmax_3":"categorical_crossentropy",
                               "weather_type_softmax_4":"categorical_crossentropy",
                               "rain_prob":'mae'},
                       metrics = self.__build_metrics()
                       )
    
  def fit(self, dataset):
    history = self.model.fit(dataset, epochs=self.params['epochs'], shuffle=True)
    return history

  def __mold_outputs(self, logits):
    outputs = []
    cat_predictions = logits[:-1]  # A batch of categorical predictions
    print(cat_predictions)
    reg_predictions = logits[-1]  # A batch of probability predictions
    # for log in logits:  # For all predictions in batch
    for cat0, cat1, cat2, cat3, cat4, reg in zip(*cat_predictions, reg_predictions):
      cats = []  # Meow
      cats = [np.argmax(cat0), np.argmax(cat1), np.argmax(cat2), np.argmax(cat3), np.argmax(cat4)]
      times = [5, 10, 15, 30, 60]
      output = {}
      for i, time in enumerate(times):
        output[str(time)] = {"type": int(cats[i]), "rain_percentage": round(reg[i].astype("float64"), 2)}
      outputs.append(output)
    return outputs

  def __mold_inputs(self, example, weather, dataset):
    """
    Formats a line from the original csv file to a Model input.
    """
    example["TIMESTAMP"] = example["TIMESTAMP"].view(np.int64)*1e-9
    example = example.drop([
                            "M_SESSION_DURATION",
                            "TIMESTAMP",
                            "M_SESSION_TIME_LEFT",
                            "M_TRACK_LENGTH",
                            "M_GAME_PAUSED",
                            "M_NUM_WEATHER_FORECAST_SAMPLES",
                            "M_TRACK_ID",
                            "M_SESSION_TYPE",
                            "M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE",
                            "M_SESSION_UID",
                            "M_PACKET_FORMAT", "M_GAME_MAJOR_VERSION",
                            "M_GAME_MINOR_VERSION", "M_PACKET_VERSION", "M_PACKET_ID",
                            "M_SESSION_TIME", "M_FRAME_IDENTIFIER", "M_PLAYER_CAR_INDEX", "M_SECONDARY_PLAYER_CAR_INDEX",
                            "M_BRAKING_ASSIST", "M_PIT_RELEASE_ASSIST", "M_ZONE_START", "M_ZONE_FLAG", "M_PIT_STOP_WINDOW_IDEAL_LAP",
                            "GAMEHOST", "M_SLI_PRO_NATIVE_SUPPORT", "M_SAFETY_CAR_STATUS", "M_ERSASSIST",
                            "M_FORMULA", "M_SEASON_LINK_IDENTIFIER", "M_PIT_ASSIST", "M_GEARBOX_ASSIST", "M_SPECTATOR_CAR_INDEX",
                            "M_PIT_STOP_WINDOW_LATEST_LAP", "M_WEEKEND_LINK_IDENTIFIER",
                            "M_DYNAMIC_RACING_LINE_TYPE", "M_PIT_STOP_REJOIN_POSITION", "M_AI_DIFFICULTY", "M_PIT_SPEED_LIMIT", "M_NETWORK_GAME", "M_TOTAL_LAPS",
                            "M_STEERING_ASSIST", "M_IS_SPECTATING", "M_DYNAMIC_RACING_LINE",
                            "M_DRSASSIST", "M_NUM_MARSHAL_ZONES",
                            "M_WEATHER_FORECAST_SAMPLES_M_WEATHER", "M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE", "M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE",
                            "Unnamed: 58"]).copy()
    typ = example['M_WEATHER']
    air = example['M_AIR_TEMPERATURE']
    vs = weather[(weather['M_WEATHER'] == typ) & (weather['M_AIR_TEMPERATURE'] == air)].drop(columns=['M_WEATHER', 'M_AIR_TEMPERATURE']).values
    out = np.hstack([[0,0], np.array(example), [0], np.squeeze(vs)])
    return tf.convert_to_tensor([dataset.pad_sequence(array=out)])


  def predict(self, x, weather):
    x = self.__mold_inputs(x, weather)
    logits = self.model.predict(x)
    molded_output = self.__mold_outputs(logits)
    return molded_output