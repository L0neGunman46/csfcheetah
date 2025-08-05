# File: src/evaluation/evaluator.py
import torch
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict

from ..models.agent import CSFAgent
from ..utils.normalizer import StateNormalizer
from .visualizer import SkillVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSFEvaluator:
    """Evaluator for trained CSF agent"""

    def evaluate(
        self,
        checkpoint_path: str,
        num_episodes: int = 10,
        num_skills: int = 8,
        max_steps: int = 1000,
    ) -> Tuple[List[List[float]], List[List[Dict]]]:
        """Evaluate trained CSF agent and compute zero-shot success rate."""

        # 1. Environment setup
        env = gym.make(self.config["environment"]["name"], render_mode="rgb_array")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # 2. Load pre-trained agent
        agent = CSFAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            skill_dim=self.config["model"]["skill_dim"],
            repr_dim=self.config["model"]["repr_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
        )
        iteration = agent.load_models(checkpoint_path)
        normalizer = StateNormalizer(state_dim)
        normalizer.load(checkpoint_path.replace(".pt", "_normalizer.pkl"))

        print(f"Evaluating CSF agent from iteration {iteration}")

        # 3. Evaluate each skill
        skill_returns: List[List[float]] = []
        skill_trajectories: List[List[Dict]] = []

        for skill_idx in range(num_skills):
            skill = agent.sample_skill()
            episode_returns: List[float] = []
            episode_trajectories: List[Dict] = []

            for ep in range(num_episodes):
                obs, _ = env.reset()
                state = obs
                trajectory: List[Dict] = []
                total_return = 0.0

                for step in range(max_steps):
                    # Normalize state
                    normalized_state = (state - state.mean()) / (
                        state.std() + 1e-8
                    )
                    state_tensor = (
                        torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
                    )
                    skill_tensor = skill.unsqueeze(0).to(device)

                    # Action
                    action = agent.policy.sample_action(
                        state_tensor, skill_tensor, noise_scale=0.0
                    )
                    action_np = action.cpu().numpy().flatten()

                    # Step environment
                    next_obs, reward, term, trunc, _ = env.step(action_np)
                    trajectory.append({"x_pos": next_obs[0]})
                    total_return += reward
                    state = next_obs

                    if term or trunc:
                        break

                episode_returns.append(total_return)
                episode_trajectories.append(trajectory)

            skill_returns.append(episode_returns)
            skill_trajectories.append(episode_trajectories)

            avg_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            print(
                f"Skill {skill_idx+1}: Return = {avg_return:.2f} ± {std_return:.2f}"
            )

        # 4. Zero-shot goal-reaching evaluation
        np.random.seed(0)
        goals = np.random.uniform(-100, 100, size=50).tolist()
        zero_shot_trajs: List[List[Dict]] = []

        for goal in goals:
            skill = agent.sample_skill()
            traj: List[Dict] = []
            obs, _ = env.reset()
            state = obs

            for step in range(max_steps):
                normalized_state = (state - state.mean()) / (state.std() + 1e-8)
                st = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
                sk = skill.unsqueeze(0).to(device)
                act = agent.policy.sample_action(st, sk, noise_scale=0.0).cpu().numpy().flatten()

                obs, _, term, trunc, _ = env.step(act)
                traj.append({"x_pos": obs[0]})

                if term or trunc:
                    break
            zero_shot_trajs.append(traj)

        success_rate = self.compute_success_rate(zero_shot_trajs, goals)
        print(f"Zero-shot success rate: {success_rate:.3f}")

        # 5. Visualize results
        self.visualizer.visualize_evaluation_results(
            skill_returns, skill_trajectories, num_skills
        )

        env.close()
        return skill_returns, skill_trajectories

    @staticmethod
    def compute_success_rate(
        trajectories: List[List[Dict]], goals: List[float], threshold: float = 0.5
    ) -> float:
        """Compute fraction of trajectories that reach within threshold of each goal."""
        successes = 0
        for traj, goal in zip(trajectories, goals):
            if not traj:
                continue
            final_pos = traj[-1]["x_pos"]
            if abs(final_pos - goal) < threshold:
                successes += 1
        return successes / len(goals) if goals else 0.0