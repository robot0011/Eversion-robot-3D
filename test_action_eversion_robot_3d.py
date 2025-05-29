import numpy as np
from eversion_robot_3d import EversionRobot3D
import matplotlib.pyplot as plt

def run_simulation():
    # Close any existing figures
    plt.close('all')
    
    # Initialize environment
    env = EversionRobot3D(obs_use=True)
    state = env.reset()
    
    # Set up interactive mode
    plt.ion()
    
    try:
        # Run simulation
        for step in range(90):
            action = 0  # Constant action (extend)
            if step > 27:
                action = 1
            if step > 30:
                action = 2
            if step > 40:
                action = 3
            if step > 60:
                action = 4
            if step > 70:
                action = 5
            if step > 80:
                action = 6
            # action = np.random.randint(0, 7)  # For random actions
            
            state, reward, done, _ = env.step(action)
            
            # Render with brief pause
            env.render()
            plt.pause(0.05)  # Slightly longer pause for better visualization
            
            # Improved status output
            pos = np.array2string(np.round(state[:3], 3), separator=', ')
            tgt = np.array2string(np.round(state[3:6], 3), separator=', ')
            print(f"Step {step+1:2d} | Action: {action} | Reward: {reward:7.2f}")
            print(f"Position: {pos}")
            print(f"Target:   {tgt}")
            print(f"length: {env.length_array}")
            print("-" * 50)
            
            if done:
                print("Termination condition reached!")
                break
                
    finally:
        # Clean up
        plt.ioff()
        plt.show()
        print("Simulation completed successfully")

if __name__ == "__main__":
    run_simulation()