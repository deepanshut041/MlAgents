<!-- <img src="./media/unity_logo.png" height="40px">
<img src="./media/pytorch_logo.jpeg" height="40px">
<img src="./media/anaconda_logo.png" height="40px">
<img src="./media/colab_logo.png" height="40px">
<img src="./media/aws_logo.png" height="40px"> -->

# :construction: This Repository is under construction. I'm currently updating the implementations with PyTorch.

# Unity ML-Agents with Pytorch

This repository contains the implementation of deep reinforcement learning algorithms to solve various unity The Environments. The deep reinforcement agents are implemented with the help of PyTorch.

<p align="center"><img src="./media/header.png" width="100%"></p>

## Index

- [**Environments**](#Environments)
  - [**Basic**](#basic)
  - [**Push Block**](#push-block)
  - [**Reacher**](#reacher)
  - [**3D Balance Ball**](#3d-balance-ball)
  - [**GridWorld**](#gridworld)
  - [**Tennis**](#tennis)
  - [**Wall Jump**](#wall-jump)
  - [**Crawler**](#crawler)
  - [**Food Collector**](#food-collector)
  - [**Hallway**](#hallway)
  - [**Bouncer**](#bouncer)
  - [**Soccer Twos**](#soccer-twos)
  - [**Walker**](#walker)
  - [**Pyramids**](#pyramids)
- [**Installation**](#installation)
- [**Any questions**](#any-questions)
- [**How to help**](#how-to-help)

## Environments

### Basic

<p align="center"><img src="./media/basic.png" height="350px"></p>

A linear movement task where the agent must move left or right to rewarding states. The goal is to move to the most reward state.

**:file_folder: [Implementation](./basic)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :heavy_check_mark:**

### Push Block

<p align="center"><img src="./media/push.png" height="350px"></p>

A platforming The Environment where the agent can push a block around. Here the goal is agent must push the block to the goal.

**:file_folder: [Implementation](./push-block)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Reacher

<p align="center"><img src="./media/reacher.png" height="350px"></p>

The Double-jointed arm which can move to target locations. Here the goal is agents must move its hand to the goal location, and keep it there.

**:file_folder: [Implementation](./reacher)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### 3D Balance Ball

<p align="center"><img src="./media/balance.png" height="350px"></p>

A balance-ball task, where the agent balances the ball on its head. Here the goal is agent must balance the ball on its head for as long as possible.

**:file_folder: [Implementation](./3d-balance-ball)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### GridWorld

<p align="center"><img src="./media/gridworld.png" height="350px"></p>

A version of the classic grid-world task. Scene contain agent, goal, and obstacles. Here the goal is an agent must navigate the grid to the goal while avoiding the obstacles.

**:file_folder: [Implementation](./gridworld)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Tennis

<p align="center"><img src="./media/tennis.png" height="350px"></p>

The two-player game where agents control rackets to hit a ball over the net. Here the goal is agents must hit the ball so that the opponent cannot hit a valid return.

**:file_folder: [Implementation](./tennis)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Wall Jump

<p align="center"><img src="./media/wall.png" height="350px"></p>

A platforming The Environment where the agent can jump over a wall. Here the goal is agent must use the block to scale the wall and reach the goal.

**:file_folder: [Implementation](./wall-jump)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Crawler

<p align="center"><img src="./media/crawler.png" height="350px"></p>

A creature with 4 arms and 4 forearms. Here the goal is agents must move their body toward the goal direction without falling.

**:file_folder: [Implementation](./crawler)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Food Collector

<p align="center"><img src="./media/foodCollector.png" height="350px"></p>

A multi-agent The Environment where agents compete to collect food. Here the goal is agents must learn to collect as many green food spheres as possible while avoiding red spheres.

**:file_folder: [Implementation](./food-collector)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Hallway

<p align="center"><img src="./media/hallway.png" height="350px"></p>

The Environment where the agent needs to find information in a room, remember it and use it to move to the correct goal. Here the goal is to move to the goal which corresponds to the color of the block in the room.

**:file_folder: [Implementation](./hallway)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Bouncer

<p align="center"><img src="./media/bouncer.png" height="350px"></p>

The Environment where the agent needs on-demand decision making. The agent must decide how to perform its next bounce only when it touches the ground. Here the goal is to catch the floating green cube. It only has a limited number of jumps.

**:file_folder: [Implementation](./bouncer)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Soccer Twos

<p align="center"><img src="./media/soccer.png" height="350px"></p>

The Environment where four agents compete in a 2 vs 2 toy soccer game. Here the goal is to get the ball into the opponent's goal while preventing the ball from entering its own goal.

**:file_folder: [Implementation](./soccer-twos)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Walker

<p align="center"><img src="./media/walker.png" height="350px"></p>

Physics-based Humanoids agents with 26 degrees of freedom. These DOFs correspond to the articulation of the following body parts: hips, chest, spine, head, thighs, shins, feet, arms, forearms and hands. Here the goal is agents must move its body toward the goal direction as quickly as possible without falling.

**:file_folder: [Implementation](./walker)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: Environment Solved :x:**

### Pyramids

<p align="center"><img src="./media/pyramids.png" height="350px"></p>

The Environment where the agent needs to press a button to spawn a pyramid, then navigate to the pyramid, knock it over, and move to the gold brick at the top. Here the goal is to move to the golden brick on top of the spawned pyramid.

**:file_folder: [Implementation](./pyramids)**

**:orange_book: [Colab](.)**

**:newspaper: [Article](.)**

**:video_camera: [Watch trained Agent](.)**

**:warning: The Environment Solved :x:**

## Installation

Create a new Python 3.7 The Environment.

```bash
conda create --name unityai python=3.7
activate unityai
```

Clone this repository locally.

```bash
git clone https://github.com/deepanshut041/ml_agents-pytorch.git
cd ml_agents-pytorch
```

Install ml-agents and other dependencies.

```bash
pip install -r requirements.txt
```

Finally, download Environments that correspond to your operating system. Copy/paste the extracted content to the folder with the name of `unity_envs`.

- [Windows (64-bit)](https://drive.google.com/open?id=1yMMMJC4ttS118eqYbr2PpDSpdnyhSoDx)

## Any questions

If you have any questions, feel free to ask me:

- **Mail**: <a href="mailto:deepanshut041@gmail.com">deepanshut041@gmail.com</a>  
- **Github**: [https://github.com/deepanshut041/Reinforcement-Learning](https://github.com/deepanshut041/Reinforcement-Learning) 
- **Website**: [https://deepanshut041.github.io/Reinforcement-Learning](https://deepanshut041.github.io/Reinforcement-Learning) 
- **Twitter**: <a href="https://twitter.com/deepanshut041">@deepanshut041</a> 

Don't forget to follow me on <a href="https://twitter.com/deepanshut041">twitter</a>, <a href="https://github.com/deepanshut041">github</a> and <a href="https://medium.com/@deepanshut041">Medium</a> to be alerted of the new articles that I publish

## How to help  

- **Clap on articles**: Clapping in Medium means that you like my articles. And the more claps I have, the more my article is shared to help them to be much more visible to the deep learning community.
- **Improve our notebooks**: if you found a bug or **a better implementation** you can send a pull request.
