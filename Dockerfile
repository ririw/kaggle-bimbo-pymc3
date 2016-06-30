FROM jupyter/datascience-notebook

RUN conda install --yes patsy
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip install git+https://github.com/pymc-devs/pymc3
RUN pip install --upgrade pip

