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

## Prediction model

## Data Augmentation

On day 1, we jumped into the challenge trying to work with the csv version of the dataset. We quickly realized that the json contained far more information (~250 sessions vs ~50 for the csv). As all our Exploratory Data Analysis pipeline was build around the csv format, we developped a small script to transform the data to a relational model. It really helped to be able to work on Oracle Cloud with huge amount of RAM to load the data in a single batch and transform it.

At the same time, we set up a small receiver to obtain more data with more weather diversity from the videogame F1 2020 using the f1-2020-telemetry package. It allowed us to run the game on certain tracks that needed more input. We mostly ran the game in the background on FP1 sessions. 

# What we brought to the Challenge

## Use of new weather dataset to improve the model engine

## Add the notion of Spatiality



