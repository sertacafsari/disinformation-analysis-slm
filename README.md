# Source Code for the Bachelor's Project: *"Hybrid Small Language Models for Accurate Multimodal Disinformation and Misinformation Analysis"*

This repository contains the source code for the Bachelor's project titled "Hybrid Small Language Models for Accurate Multimodal Disinformation and Misinformation Analysis", authored by Sertac Bahadir Afsari and supervised by Jiapan Guo. This thesis is presented to Faculty of Science and Engineering at the University of Groningen for the degree of Bachelor of Science, Computing Science.

## Abstract of the Thesis


Disinformation and misinformation are critical challenges in today's world, as they influence public opinion and pose threats to democratic institutions and right governance. With recent advancements in language models, their ability to analyze information and provide insights has significantly improved. These enhanced capabilities can be used to detect disinformation and misinformation in the social media or news sources for decreasing its influence over the public. In this study, the accuracy, effectiveness, and capabilities of Small Language Models (SLMs) and small Vision-Language Models (VLMs) for detecting and classifying disinformation and misinformation in both textual and multimodal contexts were evaluated. Consequently, several fine-tuning experiments were conducted using LIAR2, Fake News and Fauxtography datasets. In conclusion, the findings shows that SLMs and small VLMs can accurately detect and classify disinformation and misinformation. Furthermore, they may be a more efficient alternative to base LLMs used for this task.


## Setting The Environment
This project requires a Python virtual environment to run and install the necessary libraries. After creating virtual environment, it can be activated by following command:

```bash
source .venv/bin/activate
```
Then all used libraries can be installed by using the following command:
```bash
pip install -r requirements.txt
```
## Set Environmental Variables
To conduct experiments and get same results, environmental variables for Hugging Face token and Weights \& Biases token are required. When you get these, you can create an **.env** file that contains:

```python
HF_TOKEN = "<YOUR_HUGGINGFACE_TOKEN>"
WB_TOKEN = "<YOUR_WANDB_TOKEN>"
```
An example file also can be found in [.env.example file](./.env.example).

# Conduct Experiments
Experiments can be conducted by using the [scripts](./habrok_scripts/) in the Habrok. 

## Ownership & Contact
Author: Sertac Bahadir Afsari

Email: s.b.afsari@student.rug.nl
