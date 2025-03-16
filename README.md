![Logo](resources/logo.png)

---

A **Streamlit-based app** to define and run basic machine learning pipelines *quickly* and *effortlessly*.\
From data preprocessing to model training and evaluation.

![AppScreenshot](resources/img_app.png)

> [!NOTE]
> This app is based on the package [yaml-ML](https://github.com/GFaure9/yaml-ML).
> To see what are the available options for data preprocessing and model selection, please refer to the
> [yaml-ML documentation](https://gfaure9.github.io/yaml-ML/).

### 1. Installation

#### *Without Docker*

````commandline
conda create -n lazy_ml_env python=3.11
conda activate lazy_ml_env
git clone https://github.com/GFaure9/lazy-ml-app.git
cd lazy-ml-app
pip install -r requirements.txt
````

#### *With Docker* (recommended)

If you have already installed Docker on your machine, you can also install the app by building
its image:

```commandline
git clone https://github.com/GFaure9/lazy-ml-app.git
cd lazy-ml-app
docker build -t lazy-ml-app .
```

You can then check it appears in your Docker images by running `docker images`.

### 2. Usage

#### *Without Docker*

```commandline
cd lazy-ml-app
streamlit run lazyml.py
```

#### *With Docker* (recommended)

Once you have built the Docker image, in a terminal from anywhere you can run the following command to
launch the app:

```commandline
docker run --name lazy-ml-app -p 8501:8501 -v "PATH_TO_YOUR_DATA":/lazy_ml_app/data lazy-ml-app
```

> [!WARNING]
> Note that you must change `PATH_TO_YOUR_DATA` to the absolute path to the folder
> that contains the datasets files you want to work on.
> 
> Then, inside the LazyML app, the path to your dataset will have to be **the one 
> from the mounted folder**. For instance `/lazy_ml_app/data/your_dataset.csv`.
> 
> Similarly, the output folder path should be something like `/lazy_ml_app/data/outputs` 
> so that results be written to `PATH_TO_YOUR_DATA/outputs` on your machine.

**Remark**: to stop the container, run `docker stop lazy-ml-app`. To remove it,
run `docker rm lazy-ml-app` or `docker rm YOUR_CONTAINER_ID`, where `YOUR_CONTAINER_ID` can
be retrieved by looking at your containers list with the command `docker ps -a`.
