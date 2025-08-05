# File: src/evaluation/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict

from ..utils.metrics import compute_state_coverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkillVisualizer:
    """Visualization utilities for CSF"""
    
    def visualize_skills(self, agent, env, num_skills: int = 8, num_steps: int = 200):
        """Visualize learned skills"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(num_skills, 8)):
            # Sample a skill
            skill = agent.sample_skill().to(device)
            
            # Rollout with this skill
            obs, info = env.reset()
            state = obs
            
            positions = []
            for _ in range(num_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                skill_tensor = skill.unsqueeze(0)
                
                action = agent.policy.sample_action(state_tensor, skill_tensor, noise_scale=0.0)
                action = action.cpu().numpy().flatten()
                
                next_obs, _, terminated, truncated, _ = env.step(action)
                positions.append(next_obs[0]) # Still store only the x-position (float)
                
                state = next_obs
                if terminated or truncated:
                    break
            
            # Plot trajectory
            axes[i].plot(positions, alpha=0.7, linewidth=2)
            axes[i].set_title(f'Skill {i+1}')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('X Position')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_coverage_history(self, state_positions: List[float]):
        """Plot state coverage over time"""
        plt.figure(figsize=(10, 6))
        coverage_history = []
        window_size = 1000
        
        for i in range(0, len(state_positions), window_size):
            window_positions = state_positions[i:i+window_size]
            # Pass the list of floats directly to compute_state_coverage
            coverage = compute_state_coverage(window_positions) 
            coverage_history.append(coverage)
        
        plt.plot(coverage_history)
        plt.title('State Coverage Over Time')
        plt.xlabel('Time (x1000 steps)')
        plt.ylabel('Coverage (unique grid cells)')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def visualize_evaluation_results(
        self, 
        skill_returns: List[List[float]], 
        skill_trajectories: List[List[Dict]], 
        num_skills: int
    ):
        """Visualize evaluation results"""
        
        plt.figure(figsize=(12, 8))
        
        # Plot returns per skill
        plt.subplot(2, 2, 1)
        returns_data = [returns for returns in skill_returns]
        plt.boxplot(returns_data, labels=[f'Skill {i+1}' for i in range(num_skills)])
        plt.title('Returns per Skill')
        plt.ylabel('Episode Return')
        plt.xticks(rotation=45)
        
        # Plot x-position trajectories
        plt.subplot(2, 2, 2)
        colors = plt.cm.tab10(np.linspace(0, 1, num_skills))
        
        for skill_idx in range(min(num_skills, 8)):
            # Take first episode trajectory for this skill
            traj = skill_trajectories[skill_idx][0]
            x_positions = [step['x_pos'] for step in traj]
            plt.plot(x_positions, color=colors[skill_idx], 
                    label=f'Skill {skill_idx+1}', alpha=0.7, linewidth=2)
        
        plt.title('X-Position Trajectories')
        plt.xlabel('Time Steps')
        plt.ylabel('X Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot coverage comparison
        plt.subplot(2, 2, 3)
        skill_coverages = []
        for skill_idx in range(num_skills):
            positions = []
            for traj in skill_trajectories[skill_idx]:
                positions.extend([step['x_pos'] for step in traj])
            # Pass the list of floats directly to compute_state_coverage
            coverage = compute_state_coverage(positions) 
            skill_coverages.append(coverage)
        
        plt.bar(range(1, num_skills+1), skill_coverages)
        plt.title('Coverage per Skill')
        plt.xlabel('Skill')
        plt.ylabel('Coverage')
        
        # Plot skill diversity (pairwise distances)
        plt.subplot(2, 2, 4)
        skill_distances = np.zeros((num_skills, num_skills))
        
        for i in range(num_skills):
            for j in range(num_skills):
                if i != j:
                    # Compute distance between skill trajectories
                    traj_i = [step['x_pos'] for step in skill_trajectories[i][0]]
                    traj_j = [step['x_pos'] for step in skill_trajectories[j][0]]
                    
                    # Simple distance metric: difference in final positions
                    if len(traj_i) > 0 and len(traj_j) > 0:
                        skill_distances[i, j] = abs(traj_i[-1] - traj_j[-1])
        
        sns.heatmap(skill_distances, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Skill Distance Matrix')
        plt.xlabel('Skill')
        plt.ylabel('Skill')
        
        plt.tight_layout()
        plt.savefig('csf_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()