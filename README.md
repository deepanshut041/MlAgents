# Solving Unity Ml Agents with PyTorch

In this series of tutorials, we'll be solving Unity Environments with Deep Reinforcement Learning using PyTorch.  The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. 

<p align="center"><img src="./media/header.png" width="100%"></p>

Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API. Currently, unity only supports Tensorflow to train model and there is no support for PyTorch.  To train these environments using PyTorch we'll be using the standalone version of these environments.

## Index

- [**Installation**](#installation)
- [**Environments**](#Environments)
- [**Any questions**](#any-questions)
- [**How to help**](#how-to-help)

## Installation

To get started with tutorial download the repository or clone it. Than create new conda environment install required dependencies from `requirements.txt`.

- Clone this repository locally.

  ```bash
  git clone https://github.com/deepanshut041/ml_agents-pytorch.git
  cd ml_agents-pytorch
  ```

- Create a new Python 3.7 The Environment.

  ```bash
  conda create --name unityai python=3.7
  activate unityai
  ```

- Install ml-agents and other dependencies.

  ```bash
  pip install -r requirements.txt
  ```

Now our environment is ready download Standalone environments and place them in `unity_envs` folder. You can download them from below according to your operating system.

- [Windows (64-bit)](https://drive.google.com/open?id=1yMMMJC4ttS118eqYbr2PpDSpdnyhSoDx)

## Environments

<table>
  <tr>
      <td>
        <img src="./media/basic.png" width="800px">
      </td>
      <td>
      <div>
          <h3>Basic</h3>
          <p>A linear movement task where the agent must move left or right to rewarding states. The goal is to move to the most reward state.</p>
          <table style="border-collapse: collapse; border: none;">
            <tr style="border: none;background: #fff;">
              <td style="border: none;">:newspaper: <a href="https://medium.com/data-breach/" target="_blank">Article</a></td>
              <td style="border: none;">:video_camera: <a href="https://www.youtube.com/channel/UCXvsEeJqUGsiKRownM1QJqA">Video Tutorial</a></td>
            </tr>
            <tr style="border: none;background: #fff;">
              <td style="border: none;">:file_folder: <a href="./01_basic">Implementation</a></td>
              <td style="border: none;">:page_with_curl: <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">DQN</a></td>
            </tr>
          </table>
        </div>
      </td>
  </tr>
</table>

## Any questions

If you have any questions, feel free to ask me:

- **Mail**: <a href="mailto:squrlabs@gmail.com">squrlabs@gmail.com</a>  
- **Github**: [https://github.com/data-breach/MlAgents](https://github.com/data-breach/MlAgents)
- **Website**: [https://data-breach.github.io/MlAgents](https://data-breach.github.io/MlAgents)
- **Twitter**: <a href="https://twitter.com/deepanshut041">@deepanshut041</a>

Don't forget to follow me on <a href="https://twitter.com/data-breach">twitter</a>, <a href="https://github.com/data-breach">github</a> and <a href="https://medium.com/data-breach">Medium</a> to be alerted of the new articles that I publish

## How to help  

- **Clap on articles**: Clapping in Medium means that you like my articles. And the more claps I have, the more my article is shared to help them to be much more visible to the deep learning community.
- **Improve our notebooks**: if you found a bug or **a better implementation** you can send a pull request.
