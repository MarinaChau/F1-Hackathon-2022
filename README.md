# F1-Hackathon-2022
This repository is for the Hackmakers F1 competitions.
In this repository, you will find our code to predict Weather for the F1 2020 Game. We are also predicting the rain percentage.
Especially, [the notebook **deep_learning_predictions.ipynb**](Notebooks/deep_learning_predictions.ipynb) contains the prediction process.


We are participating to the 2022 edition of the hackmakers competition about Formula 1.


Formula 1 is one of the most competitive sports in the world. Engineers and technicians
from every team use weather radar screens, provided by Ubimet to the teams, which allows
them to track the current weather and make predictions during the race. Race engineers
relay precise information to drivers, including:
  * How many minutes until it starts raining
  * Intensity of the rain
  * Which corner will be hit first by the rain
  * Duration of the rain
  * Points, and even races sometimes, are won and lost based on making sense of what
the weather is going to do during a race, and being prepared as a team to act
accordingly.
Therefore, weather forecasting takes a big part on the possible outcome of a race.
Similarly, F1 2021, the official Formula 1 videogame developed by Codemasters, uses a
physics engine that behaves like the real world.

**Team Members**

Marina CHAU, Noé LALLOUET, Clément CHAUVET, Antoine BAÜMLER

# How we worked

## Exploratory Data Analysis
We first proceded with an exploration of the weather.csv dataset. You can find our analysis in the `weatherForecast_EDA` notebook.
We noticed that there were many not relevant features in our dataset, such as game parameters (GearBox Assist, Break Assist, etc).
To explore the dataset, we grouped it by sessions.

One conclusion of this exploration was that the dataset lacked of features that can help predict the weather forecast accurately (Less than 6 features). 
We also noticed that some classes were missing.

In the second time, we then decided to explore the json file. Unlike the 50 sessions found in the weather.csv, we found 250 sessions. 

For both dataset, the class repartition is very unbalanced:

Counts of items per class: {0: 376767, 1: 101197, 2: 28108, 3: 13216, 4: 4946, 5: 850}
*     % of class 0: 71.75366227117946
*      % of class 1: 19.272535441948335
*      % of class 2: 5.353048274180893
*      % of class 3: 2.516930624433424
*      % of class 4: 0.9419445269709228
*      % of class 5: 0.161878861286956

For example, the class "Storm" (5) accounts for less than 1% of the dataset.

## Data Augmentation

On day 1, we jumped into the challenge trying to work with the csv version of the dataset. We quickly realized that the json contained far more information (~250 sessions vs ~50 for the csv). As all our Exploratory Data Analysis pipeline was build around the csv format, we developped a small script to transform the data to a relational model. It really helped to be able to work on Oracle Cloud with huge amount of RAM to load the data in a single batch and transform it.

At the same time, we set up a small receiver to obtain more data with more weather diversity from the videogame F1 2020 using the f1-2020-telemetry package. It allowed us to run the game on certain tracks that needed more input. We mostly ran the game in the background on FP1 sessions. 

## Prediction model

We tested several models for the weather prediction challenge. Among our experimentations, we focused on Recurrent Neural Networks (RNNs), in the form of LSTM Networks and GRU Networks, as well as Transformer architecture.

Our best performing model is the LSTM model. It has been trained with roughly 80% of the dataset.

We are proud to report the following test set evaluation metrics:
*      Forecast at T+5 : 0.97 weather type Categorical Accuracy
*      Forecast at T+10 : 0.92 weather type Categorical Accuracy
*      Forecast at T+15 : 0.88 weather type Categorical Accuracy
*      Forecast at T+30 : 0.87 weather type Categorical Accuracy
*      Forecast at T+60 : 0.85 weather type Categorical Accuracy
*      0.088 Rain percentage Mean Average Error (mean of predictions for the 5 future timesteps)
![Alt text](/img_results/image.png "Model metrics on the test set.")

Our model correclty outputs its predictions in the form of a dictionary, as we can see below.
![Alt text](/img_results/image2.png "Prediction for a random line.")

# What we brought to the Challenge

## Use of new weather dataset to improve the model engine
As we've seen previously, it is difficult to make the prediction very accurate with few features. As an idea to improve our model, and also improve the game physics engine, we searched for other weather dataset.
Indeed, we wanted to bring some real conditions in the game, so we searched for new features such as humidity, wind bearing, wind speed, pressure, etc.

## Add the notion of Spatiality

Moreover, in formula 1, it is crucial for the driver to know the driving conditions at each corner. By driving conditions, we mean visibility, the track temperature, if it is raining, etc.
However, in the formula 1 game, the weather is the same for the whole track. It is an assumption that can be fatal in real life. Indeed, for a track as long as Spa-Francorchamps, it can rain on one corner of the track, and it can be dry on the other side. We thus decided to do a weather prediciton corner by corner, with the help of GPS coordinate. 
For each corner, we know the weather conditions, thus for each corner we will predict an accurate weather forecast.

![Alt text](/img_results/RP03.png "Rain percentage on the track")



![Alt text](/img_results/wf02.png "Weather Forecast on the track")
