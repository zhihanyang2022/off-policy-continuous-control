# Quantile Regression DQN

## Theory

**Distributional Bellman optimality operator** 

The core concept behind QR-DQN is the distributional Bellman optimality operator. The "optimality" nature of this operator is what enables the algorithm to do control; otherwise it would be just evaluating the value distribution of the current policy. This operator was analyzed *A Distributional Perspective on Reinforcement Learning* (abbreviated as DPRL), where it was proven that 

> [The] Bellman optimality operator converges, in a weak sense, to the set of optimal value distributions. However, this operator is not a contraction in any metric between distributions, and is in general much more temperamental than the policy evaluation operators. (Page 4, right column, top)

I'm not good enough in math to understand what the authors meant by "in a weak sense". Instead, I focused on the fact that this operator does indeed converge to something useful.

**Parametrization of value distribution & its effect on operators** 

Abbreviations

Parametrizing (i.e., approximation or projection) the value distribution appropriately is what enables distributional updates in practice. 

In the C51 paper:

- The projection of a distribution is another distribution (constraints: variable probability mass, fixed position) that's closest to it in KL.
- The paper showed that the DBO and DBO2 both converge.
- The paper did not show that the projected DBO2 also converges. Although it showed that DBO2 converges, projection might mess things up. The convergence of this operator is crucial for optimal control from scratch. C51's success might imply this to some extend.

In the QR-DQN paper:

- The projection of A distribution is another distribution (constraints: fixed probability mass, variable position) that's closest to it in W-1 distance. 
- The paper showed that the projected DBO converges. 
- The paper did not show that the projected DBO2 also converges. Although the C51 paper showed that DB02 converges, projection might mess things up. The convergence of this operator is crucial for optimal control from scratch. QR-DQN's success might imply this to some extend.

**Loss function and gradient.** 

In *A Distributional Perspective on Reinforcement Learning*, the authors minimizes the KL (instead of Wasserstein; but KL enables sample udpates) between the predicted and target value distributions (without proving that the projected distributional Bellman optimality operator also converges). The stuff in brackets highlight what should be but are not shown in the paper. Nevertheless, the resulting algorithm C51 performed well empirically.


