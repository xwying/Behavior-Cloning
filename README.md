# Behavior-Cloning
My solution to the behavior cloning project of Udacity's self-driving nanodegree. For detail description please see my [blog post](https://www.xiaowenying.com/self-driving/2018/08/09/behavior-cloning.html) about this project.

## Video Demos:
- Track 1: https://www.youtube.com/watch?v=4VoRetdFzRk
- Track 2: https://www.youtube.com/watch?v=JTTGcJDilx8

## File Description
- nvidia_model.py: The definition of the model structure.
- train_network.py: The main code for training the network.
- drive.py: Python scrip that load a trained model and use it to drive th car autonomously in the simulator.
- model.h5: My final saved model checkpoint.

## Usage
To reproduce the result, you will need to first download the Udacity's [self-driving simulator](https://github.com/udacity/self-driving-car-sim).

Open the simulator and enter the autonomous mode, then run the python script:

```bash
python drive.py model.h5
```

Now you should be able to see the car drives by itself.
