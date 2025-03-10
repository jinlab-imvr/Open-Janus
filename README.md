<h1 align="center"> 
    <img width="210" height="230" src="https://github.com/jinlab-imvr/Open-Janus/blob/main/openjanus.png">
</h1>



Open Janus aims to reproduce [Janus](https://github.com/deepseek-ai/Janus) from scrach.
Janus is a unified vision-language understanding and generation foundation model released by the DeepSeek team. 
Their team has released the inference code and pre-trained weights, but training code is not available.


## A Quick Overview 

 ## News
 - [TOP] We have released Open Janus, a full training pipeline to train Janus from scratch.
 - 25-03-08. This project is still quickly updating 🌝. Check TODO list to see what will be released next.

 ## Our Performance
 TBD
 ### On Benchmark
 TBD
### Visual Examples

Below is an example of Janus's output given a medical imaging prompt.  

#### Prompt: *A Chest CT image.*  

**Original Janus Output** (seems not good on medical tasks 🤔):  
![Original Janus Output](images/img_0.jpg)

 
 ## Requirement
 Refer to [Janus](https://github.com/deepseek-ai/Janus) official:
On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:
```shell
pip install -e .
```
 ## Run 

 ### Training
Training Understanding Tasks.
```shell
python train.py
```
Training Generation Tasks.
```shell
python train_generation.py
```

 ### Testing
TBD.
 ## TODO LIST

- [ ] Clean the code

 ## How to contribute

 ## Cite
 ~~~
release soon
 ~~~






