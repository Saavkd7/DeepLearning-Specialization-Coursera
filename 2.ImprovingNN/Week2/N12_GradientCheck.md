# My Diary

Today  **{{DATE_PRETTY}}**.

## General Counter
It has been    **{{DAY_SINCE_2025_08_18}}** days since I started this diary.

# Repository Counter

Day **{{DAYS_SINCE_REPO_START}}** From I started this repository


### Debugging Guide: Gradient Checking

> [!IMPORTANT]
> Use Gradient Checking **only** to verify implementation. Turn it off during actual training.

| Metric | Result | Action |
| :--- | :--- | :--- |
| **Error** $< 10^{-7}$ | Perfect | Keep going! |
| **Error** $\approx 10^{-4}$ | Suspicious | Check for small math errors. |
| **Error** $> 10^{-2}$ | Bug | Check `dW`, `db`, and `dZ` logic. |

#### Implementation Note
```python
# Pseudo-code for one parameter
theta_plus = theta + epsilon
theta_minus = theta - epsilon
grad_approx = (J(theta_plus) - J(theta_minus)) / (2 * epsilon)y

