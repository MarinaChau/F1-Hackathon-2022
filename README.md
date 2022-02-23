# F1-Hackathon-2022
This repository is for the Hackmakers F1 competitions.
In this repository, you will find our code to predict Weather for the F1 2020 Game. We are also predicting the rain percentage.


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

## Prediction model

## Data Augmentation

# What we brought to the Challenge

## Use of new weather dataset to improve the model engine
As we've seen previously, it is difficult to make the prediction very accurate with few features. As an idea to improve our model, and also improve the game physics engine, we searched for other weather dataset.
Indeed, we wanted to bring some real conditions in the game, so we searched for new features such as humidity, wind bearing, wind speed, pressure, etc.

## Add the notion of Spatiality

![Alt text](../F1-Hackathon-2022/img_results/RP03.png "Title")

