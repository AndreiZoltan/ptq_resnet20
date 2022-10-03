import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
   name="ptq_resnet20",
   version="0.1",
   package_data={"handmade_ptq": ["py.typed"]},
   long_description=read('README'),
)
