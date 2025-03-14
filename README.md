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

#### Prompt: *A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue..*  

**Output** :  
![Output](images/img_0.jpg)

#### Prompt: *What is this meme trying to express*
 **Image** :
 ![Output](images/doge.png)
**Output** : *Assistant: The meme is trying to express the difference between two types of neural networks: "Decoupling Visual Encoding" and "Single Visual Encoder." The image uses the Shiba Inu (a breed of dog) to visually represent the two concepts. The "Decoupling Visual Encoding" is depicted as a muscular, strong dog, while the "Single Visual Encoder" is shown as a smaller, less muscular dog sitting down. This visual metaphor is used to contrast the two approaches to neural network architecture, highlighting the idea that a more powerful and efficient system is needed for visual data processing.*

 
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






