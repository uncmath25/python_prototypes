FROM jupyter/datascience-notebook:65761486d5d3

USER jovyan
RUN mkdir -p /home/jovyan/.ipython/profile_default

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
