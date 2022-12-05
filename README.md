# Reinforcement-learning
This repository contains code implementations of basic RL algorithms.

Run the command below to install dependencies needed.

```bash
pip install -r requirements.txt
```

Run the commands below to install packages required.

```bash
sudo add-apt-repository universe

sudo xargs -a packages.txt apt-get install
```
The error `libGL error: MESA-LOADER: failed to open iris` can be solved using the instructions found [here.](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris)


### Installing ROMs.
***
Download the ROMS [here](http://www.atarimania.com/roms/Roms.rar) and extract the `.rar` file. 
Run `python -m atari_py.import_roms <path to folder>`. The ROMs should be copied to your `atari_py` installation directory.
