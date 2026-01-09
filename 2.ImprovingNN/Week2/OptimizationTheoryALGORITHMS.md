# My Diary

Today  **{{DATE_PRETTY}}**.

## General Counter
It has been    **{{DAY_SINCE_2025_08_18}}** days since I started this diary.

# Repository Counter

Day **{{DAYS_SINCE_REPO_START}}** From I started this repository


# 🚀 Deep Learning Optimizers: From Momentum to Adam

A concise guide to understanding how modern optimizers navigate the loss landscape.

---

## 1. Momentum: The Snowball ❄️
**Momentum** accumulates the gradient of past steps to determine the direction of the next move, effectively smoothing out oscillations.

**Mathematical Logic:**
$$v_t = \beta_1 v_{t-1} + (1 - \beta_1) \nabla J(\theta)$$
$$\theta = \theta - \alpha v_t$$

* **Role:** Provides **inertia** to push through flat regions (plateaus) and local minima.
* **Intuition:** It remembers where it was going.

---

## 2. RMSprop: The Shock Absorber 🏎️
**Root Mean Squared Propagation** scales the learning rate for each parameter by the running average of recent gradient magnitudes.

**Mathematical Logic:**
$$s_t = \beta_2 s_{t-1} + (1 - \beta_2) (\nabla J(\theta))^2$$
$$\theta = \theta - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla J(\theta)$$

* **Role:** Dampens **vertical oscillations** to speed up horizontal progress.
* **Intuition:** It brakes if the road is too bumpy and accelerates if it's smooth.

---

## 3. Adam: The Intelligent Driver 🤖
**Adaptive Moment Estimation** is the "Golden Child." It combines the direction of **Momentum** with the stability of **RMSprop**, plus a **Bias Correction** to handle the start of training.

**The Unification:**
1. **First Moment (Momentum):** $v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$
2. **Second Moment (RMSprop):** $s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$
3. **Bias Correction:** $\hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$

**Update Rule:**
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{s}_t} + \epsilon} \hat{v}_t$$



---

## Summary Table

| Feature | Momentum | RMSprop | Adam |
| :--- | :--- | :--- | :--- |
| **Focus** | Direction/Inertia | Stability/Scale | Direction + Stability |
| **Parameter** | $\beta_1$ (usually 0.9) | $\beta_2$ (usually 0.999) | Both $\beta_1$ & $\beta_2$ |
| **Adaptivity** | No | Yes (per-parameter) | Yes (Highly adaptive) |

> **Pro Tip:** Use **Learning Rate Decay** as a global "speed limit" that decreases over time ($t$) to ensure the model settles perfectly into the global minimum.
