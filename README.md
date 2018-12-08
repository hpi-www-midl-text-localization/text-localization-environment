# text-localization-environment
The environment we will use to train our text localization agent.

For now, we will stick to the structure of the environment as laid out in the 
[base paper](http://slazebni.cs.illinois.edu/publications/iccv15_active.pdf). We conform to the 
[OpenAi](https://github.com/openai/gym)-Interface.

## Installation

To install the package, clone this repository, open a terminal in the top folder of the package and call:
```
pip install -e .
```

To unistall call 

```
pip uninstall text_localization_environment
```

To use the environment, call

```
from text_localization_environment import TextLocEnv
```
