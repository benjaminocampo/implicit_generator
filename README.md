# [Playing the Part of the Sharp Bully: Generating Adversarial Examples for Implicit Hate Speech Detection](https://aclanthology.org/2023.findings-acl.173/)
This repository contains the generation and implementation details of the paper "Playing the Part of the Sharp Bully: Generating Adversarial Examples for
Implicit Hate Speech Detection" accepted at Findings of ACL 2023.

# Index

- [Installation](#installation)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Ethical Statements](#ethical-statements)
- [Contributing](#contributing)
- [Cite us](#cite-us)

# Installation

First, make sure that `conda` is installed on your machine. You can check if it
is installed by running `conda --version` in your terminal.

Clone the repository that contains of this project onto your local machine.

Once conda is installed, navigate to the directory where the application is
located in the terminal.

Run the command `make create_environment` to create a new conda environment for
your application.

Then, run the command `make requirements` to install the necessary dependencies
for your application in the newly created environment.

Optionally, you can test the environment by running the command `make
test_environment`

Once the environment is created and the dependencies are installed, you should
be able to run the experiments.

# Dataset

We are using the ISHate dataset (An In-depth Analysis of Implicit and Subtle
Hate Speech Messages: https://aclanthology.org/2023.eacl-main.147/) split into
train, dev, and test. It can be found in the directory `./data/` as csv files.
They can be easily opened with `pandas`:

```python
import pandas as pd

train = pd.read_parquet("./data/ishate_train.csv")
dev = pd.read_parquet("./data/ishate_train.csv")
test = pd.read_parquet("./data/ishate_train.csv")
```

Alongside the ISHate dataset, we have also compiled a lexicon tailored for this
dataset named `ishate_lexicon.csv`, which resides in the `./data/` directory.
This lexicon can be used to better understand the data and weight generation.

Here's how to load it with `pandas`:

```python
lexicon = pd.read_csv("./data/ishate_lexicon.csv")
```

# Experiments

In the experiments directory, you'll find the scripts responsible for generating
implicit messages. A configuration file is included that lists all necessary
parameters for running the code. It's important to note that these scripts
utilize the OpenAI GPT-3 models, requiring a model secret key to access. Users
must manage their own keys and adhere to OpenAI's usage policies and
restrictions.

Models and results are registered using MLFlow in the same directory. In order
to display them, you can use the mlflow ui by running on the shell

```shell
mlflow ui
```

And opening your localhost in the port 5000: [http://127.0.0.1:5000/](http://127.0.0.1:5000/). For more information on MLFlow read [https://mlflow.org/](https://mlflow.org/)

# Ethical Statements

This paper includes examples of hate speech (HS) from established linguistic
resources for HS detection, which do not represent the authors' viewpoints. Our
intent is to mitigate and filter HS from social media resources. Nonetheless,
there is a potential risk of misuse, as the methods described could be used to
guide large language models to generate subtle and implicit hate speech. We
believe that developing robust classifiers and new methods for data creation and
collection is vital to explore and address implicit and subtle online hate
speech effectively and to prevent the spread of such harmful content. Our work
is a step towards this goal and we encourage the scientific community to delve
into these issues.

# Contributing

We are thrilled that you are interested in contributing to our work! Your
contributions will help to make our project even better and more useful for the
community.

Here are some ways you can contribute:

- Bug reporting: If you find a bug in our code, please report it to us by
  creating a new issue in our GitHub repository. Be sure to include detailed
  information about the bug and the steps to reproduce it.

- Code contributions: If you have experience with the technologies we are using
  and would like to contribute to the codebase, please feel free to submit a
  pull request. We welcome contributions of all sizes, whether it's a small bug
  fix or a new feature.

- Documentation: If you find that our documentation is lacking or could be
  improved, we would be grateful for your contributions. Whether it's fixing
  typos, adding new examples or explanations, or reorganizing the information,
  your help is greatly appreciated.

- Testing: Testing is an important part of our development process. We would
  appreciate it if you could test our code and let us know if you find any
  issues.

- Feature requests: If you have an idea for a new feature or improvement, please
  let us know by creating a new issue in our GitHub repository.

Please note that I am committed to updating and maintaining this repository to
ensure the community has access to our ongoing work. However, due to the volume
of requests and the nature of research, response times may vary. I ask for your
patience and understanding as I work to address issues and collaborate with
fellow researchers.

All contributions are welcome and appreciated! We look forward to working with
you to improve our project.

# Cite us

```tex
@inproceedings{ocampo-etal-2023-playing,
    title = "Playing the Part of the Sharp Bully: Generating Adversarial Examples for Implicit Hate Speech Detection",
    author = "Ocampo, Nicolas  and
      Cabrio, Elena  and
      Villata, Serena",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.173",
    doi = "10.18653/v1/2023.findings-acl.173",
    pages = "2758--2772",
    abstract = "Research on abusive content detection on social media has primarily focused on explicit forms of hate speech (HS), that are often identifiable by recognizing hateful words and expressions. Messages containing linguistically subtle and implicit forms of hate speech still constitute an open challenge for automatic hate speech detection. In this paper, we propose a new framework for generating adversarial implicit HS short-text messages using Auto-regressive Language Models. Moreover, we propose a strategy to group the generated implicit messages in complexity levels (EASY, MEDIUM, and HARD categories) characterizing how challenging these messages are for supervised classifiers. Finally, relying on (Dinan et al., 2019; Vidgen et al., 2021), we propose a {``}build it, break it, fix it{''}, training scheme using HARD messages showing how iteratively retraining on HARD messages substantially leverages SOTA models{'} performances on implicit HS benchmarks.",
}

```
