# Quantile Regression DQN

## 5-minute theory

**Distributional Bellman optimality operator** 

The core concept behind QR-DQN is the distributional Bellman optimality operator. The "optimality" nature of this operator is what enables the algorithm to do control; otherwise it would be just evaluating the value distribution of the current policy. This operator was analyzed *A Distributional Perspective on Reinforcement Learning*, where it was proven that 

> [The] Bellman optimality operator converges, in a weak sense, to the set of optimal value distributions. However, this operator is not a contraction in any metric between distributions, and is in general much more temperamental than the policy evaluation operators. (Page 4, right column, top)

I'm not good enough in math to understand what the authors meant by "in a weak sense". Instead, I focused on the fact that this operator does indeed converge to something useful.

**Parametrization of value distribution** 

Parametrizing (i.e., approximation or projection) the value distribution appropriately is what enables distributional updates in practice. 

**Loss function.** 

In *A Distributional Perspective on Reinforcement Learning*, the authors minimizes the KL (instead of Wasserstein; but KL enables sample udpates) between the predicted and target value distributions (without proving that the projected distributional Bellman optimality operator also converges). The stuff in brackets highlight what should be but are not shown in the paper. Nevertheless, the resulting algorithm C51 performed well empirically.


