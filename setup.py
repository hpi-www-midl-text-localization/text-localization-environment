from setuptools import setup

requirements = [
      'gym',
      'chainer',
      'numpy',
      'Pillow']

setup(name='text_localization_environment',
      version='0.0.1',
      install_requires=requirements  # And any other dependencies foo needs
)