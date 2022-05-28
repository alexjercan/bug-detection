# Bug Detection and Repair

This repository contains the script used for dataset generation
[codenet.py](codenet.py). We have also added some exploratory data analysis
notebooks for the generated dataset. The `docker-example` folder contains the
demo for the bug detection and repair pipeline and uses models trained on the
codenetpy dataset. The demo application is created using the Streamlit python
library and runs on localhost. Finally we have included a REST API endpoint, in
`rest-api` that can be used to deploy a backend for a custom frontend
application. We have used Flask and gunicorn for deployment inside Docker.

![thumbnail](./resources/architecture.png)

The generated dataset and the notebooks used to train the model can be found on
Kaggle. Dataset can be found
[here](https://www.kaggle.com/datasets/alexjercan/codenetpy)
