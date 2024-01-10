from setuptools import setup

setup(name='serotina',
      version='1.0',
      description='A simple semi-analytic model for the evolution of supermassive black holes over cosmic time.',
      url='http://github.com/ARRicarte/serotina/',
      author='ARRicarte',
      author_email='angelo.ricarte@cfa.harvard.edu',
      license='GPLv3',
      packages=['serotina']
      install_requires=['numpy','scipy','matplotlib','cython'])
