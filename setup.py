#!/usr/bin/env python

from distutils.core import setup

setup(name='robot-fabrik-cga',
      version='0.0.1',
      description='Fabrik Solver for robotics using Conformal Geometric Algebra',
      author='Pau Carre Cardona',
      author_email='pau.carre@gmail.com',
      url='https://github.com/paucarre/robot-fabrik-cga',
      packages=['fabrik'],
      package_dir = {'fabrik': 'src'},
     )
