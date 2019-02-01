# text-localization-environment

[![Build Status](https://travis-ci.com/hpi-www-midl-text-localization/text-localization-environment.svg?branch=master)](https://travis-ci.com/hpi-www-midl-text-localization/text-localization-environment)

The environment we will use to train our text localization agent.

For now, we will stick to the structure of the environment as laid out in the 
[base paper](http://slazebni.cs.illinois.edu/publications/iccv15_active.pdf). We conform to the 
[OpenAi](https://github.com/openai/gym)-Interface.

## Installation

To install the package, clone this repository, open a terminal in the top folder of the package and call:
```
pip install -e .
```

To uninstall call 

```
pip uninstall text_localization_environment
```

To use the environment, call

```
from text_localization_environment import TextLocEnv
```

To download the ResNet-152 caffemodel (it isn't downloaded automatically) see [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
and save it where necessary (an error will tell you where if you try to create a TextLocEnv).