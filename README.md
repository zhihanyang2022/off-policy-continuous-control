CITE image

PLEASE NOTE THAT THIS REPO IS UNDER ACTIVE CONSTRUCTION.

Table of content

# CleanRL 🧚‍♂️ 

*Minimalistic, well-documented implementation of model-free off-policy deep RL algorithms using PyTorch.*

*This repo implements all algorithms attached to the "Q-learning" node in the diagram below.*

*This project was inpsired by OpenAI Spinning Up, which helped me tremendously along the way.*

<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width=600>
</p>

## Features

<table>
<thead>
<tr>
<th style="text-align:center">Problems with some repos</th>
<th style="text-align:center">Solutions</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">&quot;Code works but I don&#39;t understand the how&quot;</td>
<td style="text-align:center">Offers docs and implementation notes</td>
</tr>
<tr>
<td style="text-align:center">Uses sophisticated abstraction</td>
<td style="text-align:center">Offers graphical explanation of design choices</td>
</tr>
<tr>
<td style="text-align:center">Does not have a good requirements file</td>
<td style="text-align:center">Has a requirements file tested across multiple machines</td>
</tr>
<tr>
<td style="text-align:center">Does not compare against authoritative repos</td>
<td style="text-align:center">Compares against OpenAI Spinning Up*</td>
</tr>
<tr>
<td style="text-align:center">Does not test on many environments</td>
<td style="text-align:center">Tests on many tasks including Atari &amp; Mujoco</td>
</tr>
</tbody>
</table>


\* However, not all algorithms here are implemented in OpenAI Spinning Up.

## Codebase design

The diagrams below are created using Lucidchart.

### Overview

<p align="center">
  <img src="design.svg" width=600>
</p>

### Abstract classes

## Implemented algorithms and notes

Q-learning algorithms:
- Deep Q-learing
- Categorical 51
- <a target="_blank" href="https://nbviewer.jupyter.org/github/zhihanyang2022/CleanRL/blob/main/notes/qrdqn.pdf" type="application/pdf">Quantile-Regression DQN</a>

Policy optimization algorithms:
- TODO

Both:
- TODO

## Gallery of GIFs of learned policies
