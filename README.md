
# FlappyBird with DQN
Tensorflow implementation for training a network to play FlappyBird

<img src="./assets/result.gif" width="288">

This repository contains train,test codes and saved model for reproduce.

--------------------------
## Prerequisites
- Tensorflow
- python-pygame

## Getting Started
### Installation
- Install tensorflow from https://www.tensorflow.org/install/
- Clone this repo:
```bash
git clone https://github.com/codeslake/flappybird_DQN.git
cd flapybird_DQN
```
- Install python-pygame
```bash
sudo apt-get install python-pygame
```

## Training and Test Details
- you need to specify directories for checkpoint and buffer saving directory in config.py
- To train a model,  
```bash
python main.py --is_Train True
```
- To test the model,
```bash
python main.py --is_Train False
```
- To render and to show the plot, add
```bash
python main.py --is_Train True --render True --plot True
```

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition, personalization technology through user preference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)
