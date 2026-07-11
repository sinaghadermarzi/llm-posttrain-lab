I want to scaffold an experiment or a series of experiments that shows why we use reinforcement learning (PPO and GRPO) for preference alignment in post-training of LLMS, rather than simple SFT or DPO. SFT (based on Next Token Prediction loss) is usually considered an imitation learning approach, while reinforcement learning approaches are known to be preserving the original distribution / learning of LLM but inducing preference over certain answers using the KL divergence loss term. 
- We probably should pick some public dataset  and base model for some task that can demonstrate the difference between these approaches. The model / dataset should be simple but demonstrative
- This will be probably run on a node with 4 x A100 80GB GPUs preferably for up to few days. So you can adjust the size of the model and experiment accordingly.

Fee free to ask any question you may have 

![[ss1.png]]

![[ss2.png]]

![[ss3.png]]
