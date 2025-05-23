# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
## POLICY EVALUATION FUNCTION
Policy evaluation refers to the objective and systematic examination of the effects of ongoing policies and public programs on their intended goals. It involves assessing whether policies are achieving their stated objectives and identifying any impediments to their attainment.
![image](https://github.com/user-attachments/assets/a895dad5-6784-4ec0-8da0-b3f254ef59d6)
# Program
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        H = 0
        for s in range(len(P)):
            v = V[s]
            new_v = 0
            action = pi(s)
            for prob, next_state, reward, done in P[s][action]:
                new_v += prob * (reward + gamma * V[next_state])
            V[s] = new_v
            H = max(H, abs(v - V[s]))
        if H < theta:
            break
    return V
```

## OUTPUT:
# policy 1:
![image](https://github.com/user-attachments/assets/4b4d2c10-c0b3-4a28-9daf-436aa4119645)
# policy 2:
![image](https://github.com/user-attachments/assets/ce30dd5c-4225-4675-b925-a504317f643d)
# State-Value function 1:
![image](https://github.com/user-attachments/assets/ba311968-0918-46c5-a9f6-aead23791579) 
# State-Value function 2:
 ![image](https://github.com/user-attachments/assets/4607dfd5-b1a8-46e8-a5a3-1f67b7bfc17d)
# comparison of policy 1 and policy 2
![image](https://github.com/user-attachments/assets/59dd0489-f143-4251-ac7a-38146b0e80a6)

## RESULT:
Thus, The Python program to evaluate the given policy is successfully executed.
