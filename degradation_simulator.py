import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig
from scipy.optimize import linprog
import pandas as pd
import random
import time


class MaintenanceOptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimal Maintenance Intervention Simulator")
        self.root.geometry("1280x800")
        
        # Initialize system parameters from the paper
        self.C = tk.IntVar(value=3)  # Number of components
        self.K = tk.IntVar(value=3)  # Maximum deterioration level
        self.alpha = tk.DoubleVar(value=0.25)  # Prob. of degradation (1-alpha in paper)
        self.simulation_steps = tk.IntVar(value=100)
        self.yellow_threshold = tk.IntVar(value=5)  # Maximum acceptable yellow states
        
        # Cost parameters from the paper
        self.c1 = tk.DoubleVar(value=100)  # Preventive maintenance cost
        self.c2 = tk.DoubleVar(value=200)  # Corrective maintenance cost
        self.ct = tk.DoubleVar(value=30)   # Transfer cost per component
        self.cr = tk.DoubleVar(value=50)   # Replacement cost per component
        self.cs = tk.DoubleVar(value=60)   # Shortage cost per component
        self.ce = tk.DoubleVar(value=30)   # Excess cost per component
        
        # Component-specific parameters
        self.component_params = []  # Will hold parameters for each component
        
        # Results placeholders
        self.optimal_policy = None
        self.simulation_results = None
        self.maintenance_events = []
        
        # Create the main interface
        self.create_notebook()
        self.create_parameter_frame()
        self.create_cost_frame()
        self.create_action_frame()
        self.create_results_frame()
        
        # Initialize component parameters
        self.update_component_params()

    def create_notebook(self):
        """Create the main tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create main tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.policy_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Setup & Run")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.policy_tab, text="Optimal Policy")
        
        # Configure the visualization tab with sub-tabs
        self.viz_notebook = ttk.Notebook(self.visualization_tab)
        self.viz_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create visualization sub-tabs
        self.time_series_tab = ttk.Frame(self.viz_notebook)
        self.component_heatmap_tab = ttk.Frame(self.viz_notebook)
        self.cost_analysis_tab = ttk.Frame(self.viz_notebook)
        self.signal_history_tab = ttk.Frame(self.viz_notebook)
        
        self.viz_notebook.add(self.time_series_tab, text="Component States")
        self.viz_notebook.add(self.component_heatmap_tab, text="Degradation Heatmap")
        self.viz_notebook.add(self.cost_analysis_tab, text="Cost Analysis")
        self.viz_notebook.add(self.signal_history_tab, text="Signal History")

    def create_parameter_frame(self):
        """Create the system parameters frame"""
        param_frame = ttk.LabelFrame(self.setup_tab, text="Simulation Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # System parameters
        ttk.Label(param_frame, text="Number of Components (C):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        c_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.C, width=5)
        c_spinbox.grid(row=0, column=1, padx=5, pady=5)
        c_spinbox.bind("<Return>", self.update_component_params)
        c_spinbox.bind("<<Increment>>", self.update_component_params)
        c_spinbox.bind("<<Decrement>>", self.update_component_params)
        
        ttk.Label(param_frame, text="Maximum Deterioration Level (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        k_spinbox = ttk.Spinbox(param_frame, from_=2, to=15, textvariable=self.K, width=5)
        k_spinbox.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Degradation Probability (1-α):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        alpha_spinbox = ttk.Spinbox(param_frame, from_=0.05, to=0.95, increment=0.05, textvariable=self.alpha, width=5)
        alpha_spinbox.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Yellow Signal Threshold:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(param_frame, from_=1, to=20, textvariable=self.yellow_threshold, width=5).grid(
            row=3, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Simulation Steps:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(param_frame, from_=50, to=1000, increment=50, textvariable=self.simulation_steps, width=5).grid(
            row=4, column=1, padx=5, pady=5)
        
        # Button to edit component-specific parameters
        ttk.Button(param_frame, text="Edit Component Parameters", command=self.open_component_editor).grid(
            row=5, column=0, columnspan=2, padx=5, pady=10)

    def create_cost_frame(self):
        """Create the cost parameters frame"""
        cost_frame = ttk.LabelFrame(self.setup_tab, text="Cost Parameters")
        cost_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        
        # Fixed costs
        ttk.Label(cost_frame, text="Preventive Maintenance Cost (c₁):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=50, to=500, increment=10, textvariable=self.c1, width=6).grid(
            row=0, column=1, padx=5, pady=5)
        
        ttk.Label(cost_frame, text="Corrective Maintenance Cost (c₂):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=100, to=1000, increment=10, textvariable=self.c2, width=6).grid(
            row=1, column=1, padx=5, pady=5)
        
        # Variable costs
        ttk.Label(cost_frame, text="Transfer Cost per Component (cₜ):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=10, to=100, increment=5, textvariable=self.ct, width=6).grid(
            row=2, column=1, padx=5, pady=5)
        
        ttk.Label(cost_frame, text="Replacement Cost per Component (cᵣ):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=20, to=200, increment=5, textvariable=self.cr, width=6).grid(
            row=3, column=1, padx=5, pady=5)
        
        ttk.Label(cost_frame, text="Shortage Cost per Component (cₛ):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=20, to=200, increment=5, textvariable=self.cs, width=6).grid(
            row=4, column=1, padx=5, pady=5)
        
        ttk.Label(cost_frame, text="Excess Cost per Component (cₑ):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=10, to=100, increment=5, textvariable=self.ce, width=6).grid(
            row=5, column=1, padx=5, pady=5)

    def create_action_frame(self):
        """Create the action buttons frame"""
        action_frame = ttk.Frame(self.setup_tab)
        action_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")
        
        ttk.Button(action_frame, text="Calculate Optimal Policy", command=self.calculate_optimal_policy, width=25).grid(
            row=0, column=0, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Run Simulation", command=self.run_simulation, width=25).grid(
            row=1, column=0, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Reset", command=self.reset_simulation, width=25).grid(
            row=2, column=0, padx=5, pady=5)

    def create_results_frame(self):
        """Create the results display frame"""
        results_frame = ttk.LabelFrame(self.setup_tab, text="Simulation Results")
        results_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        
        # Status and progress
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200)
        self.progress_bar.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Key metrics frame
        metrics_frame = ttk.LabelFrame(results_frame, text="Key Performance Metrics")
        metrics_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Metrics labels
        self.create_metric_label(metrics_frame, "Total Cost:", 0)
        self.create_metric_label(metrics_frame, "Uptime Percentage:", 1)
        self.create_metric_label(metrics_frame, "Mean Time Between Failures:", 2)
        self.create_metric_label(metrics_frame, "Number of Interventions:", 3)
        self.create_metric_label(metrics_frame, "Preventive Maintenance Count:", 4)
        self.create_metric_label(metrics_frame, "Corrective Maintenance Count:", 5)
        self.create_metric_label(metrics_frame, "Yellow Signal Threshold Reached:", 6)
        
        # Configure grid expansion
        self.setup_tab.columnconfigure(1, weight=1)
        
        # Results log
        log_frame = ttk.LabelFrame(results_frame, text="Simulation Log")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_metric_label(self, parent, text, row):
        """Helper method to create metric labels with consistent formatting"""
        ttk.Label(parent, text=text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
        label = ttk.Label(parent, text="N/A")
        label.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        setattr(self, f"{text.lower().replace(' ', '_').replace(':', '')}_label", label)

    def update_component_params(self, event=None):
        """Update component parameters when number of components changes"""
        num_components = self.C.get()
        
        # Reset the component_params list with the current number of components
        # Preserve existing values if possible
        old_params = self.component_params.copy() if hasattr(self, 'component_params') and self.component_params else []
        self.component_params = []
        
        for i in range(num_components):
            if i < len(old_params):
                # Keep existing parameters
                self.component_params.append(old_params[i])
            else:
                # Create new parameters with default values
                self.component_params.append({
                    'name': f"Component {i+1}",
                    'k': self.K.get(),  # Maximum deterioration level
                    'p': 1 - self.alpha.get(),  # Degradation probability
                    'current_state': 0  # Initial state
                })

    def open_component_editor(self):
        """Open a window to edit component parameters"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Component Parameters Editor")
        editor_window.geometry("600x400")
        editor_window.grab_set()  # Make the window modal
        
        # Create a frame with scrollbar
        main_frame = ttk.Frame(editor_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add a canvas with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Headers
        ttk.Label(scrollable_frame, text="Component Name", font=("Arial", 10, "bold")).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Failure Threshold (K)", font=("Arial", 10, "bold")).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Degradation Prob (P)", font=("Arial", 10, "bold")).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Component parameter fields
        name_vars = []
        k_vars = []
        p_vars = []
        
        for i, comp in enumerate(self.component_params):
            # Variables to hold values
            name_var = tk.StringVar(value=comp['name'])
            k_var = tk.IntVar(value=comp['k'])
            p_var = tk.DoubleVar(value=comp['p'])
            
            name_vars.append(name_var)
            k_vars.append(k_var)
            p_vars.append(p_var)
            
            # Create entry fields
            ttk.Entry(scrollable_frame, textvariable=name_var).grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Spinbox(scrollable_frame, from_=1, to=self.K.get(), textvariable=k_var, width=5).grid(
                row=i+1, column=1, padx=5, pady=2)
            ttk.Spinbox(scrollable_frame, from_=0.01, to=1.0, increment=0.01, textvariable=p_var, width=5).grid(
                row=i+1, column=2, padx=5, pady=2)
        
        # Save button
        def save_parameters():
            for i in range(len(self.component_params)):
                self.component_params[i]['name'] = name_vars[i].get()
                self.component_params[i]['k'] = k_vars[i].get()
                self.component_params[i]['p'] = p_vars[i].get()
            editor_window.destroy()
        
        ttk.Button(editor_window, text="Save Changes", command=save_parameters).pack(pady=10)

    def log_message(self, message):
        """Add a message to the log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Scroll to the end

    def calculate_optimal_policy(self):
        """Calculate the optimal maintenance intervention policy"""
        self.status_label.config(text="Calculating optimal policy...", foreground="blue")
        self.log_message("Starting optimal policy calculation")
        
        # Get system parameters
        C = self.C.get()
        K = self.K.get()
        alpha = self.alpha.get()  # Probability of NOT degrading
        yellow_threshold = self.yellow_threshold.get()
        
        # Update progress
        self.progress_var.set(10)
        self.root.update()
        
        # Since we are using a direct approach based on thresholds rather than calculating
        # an optimal policy through an MDP or other optimization method,
        # we'll create a simple policy dictionary
        policy = {}
        
        # Our policy is:
        # 1. For red signal (2), always intervene immediately with all components
        # 2. For yellow signal (1), intervene after yellow_threshold consecutive yellows
        # 3. For green signal (0), never intervene
        
        # This approach follows the maintenance rule:
        # - A red signal (component failure) triggers immediate maintenance
        # - A yellow threshold being exceeded triggers preventive maintenance
        
        self.log_message(f"Creating policy with yellow threshold = {yellow_threshold}")
        
        # Create a simplified version of the policy
        max_steps = 200  # A reasonable upper limit for time steps
        
        # Create the policy dictionary
        for t in range(max_steps):
            # For red signal, always take all components for maintenance
            policy[(t, 2)] = C
            
            # For yellow signal, the decision depends on the threshold
            # But for simplicity in our policy representation, we'll use the original
            # policy mapping model. The actual threshold counting happens in simulation.
            policy[(t, 1)] = 0  # Default is no intervention
            
            # For green signal, never intervene
            policy[(t, 0)] = 0
            
        self.optimal_policy = policy
        
        # Update the policy display
        self.display_maintenance_rules()
        
        # Complete the progress bar
        self.progress_var.set(100)
        self.status_label.config(text="Maintenance rules set", foreground="green")
        self.log_message("Maintenance rules have been set based on thresholds")

    def display_maintenance_rules(self):
        """Display the maintenance rules in the policy tab"""
        # Clear existing content
        for widget in self.policy_tab.winfo_children():
            widget.destroy()
        
        # Create a frame for the rules display
        frame = ttk.Frame(self.policy_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create title
        ttk.Label(frame, text="Maintenance Rules", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create rules description
        rules_text = tk.Text(frame, wrap="word", height=15, width=80)
        rules_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Insert the rules with formatting
        rules_content = f"""
The system follows these maintenance rules:

1. RED SIGNAL (Failure Detection):
   - If ANY component reaches its maximum deterioration level (K={self.K.get()}), 
     the system emits a RED signal.
   - Immediate maintenance is performed, restoring all components to perfect condition.
   - All components ({self.C.get()} total) are taken for the maintenance operation.

2. YELLOW SIGNAL (Degradation Detection):
   - If ANY component is degraded but none have failed, the system emits a YELLOW signal.
   - If the system remains in YELLOW state for {self.yellow_threshold.get()} consecutive time steps,
     preventive maintenance is performed.
   - The yellow counter resets after each maintenance operation.

3. GREEN SIGNAL (Perfect Condition):
   - When ALL components are in perfect condition, the system emits a GREEN signal.
   - No maintenance action is taken in this state.

Cost Parameters:
- Preventive Maintenance (Yellow-triggered): {self.c1.get()} units
- Corrective Maintenance (Red-triggered): {self.c2.get()} units
- Transfer Cost per Component: {self.ct.get()} units
- Replacement Cost per Component: {self.cr.get()} units
- Shortage Cost per Component: {self.cs.get()} units
- Excess Cost per Component: {self.ce.get()} units

Degradation Model:
- Each component has a {(1-self.alpha.get())*100:.1f}% chance to degrade by one level each time step.
- Components degrade independently of each other.
"""
        
        rules_text.insert("1.0", rules_content)
        rules_text.config(state="disabled")  # Make text read-only
        
        # Add a visual representation of the policy
        canvas_frame = ttk.LabelFrame(frame, text="Policy Visualization")
        canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        fig = mpl_fig.Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Draw the state machine
        ax.plot([0, 1], [0, 1], 'go-', markersize=15, label='Green')
        ax.plot([1, 2], [1, 1], 'yo-', markersize=15, label='Yellow')
        ax.plot([2, 3], [1, 0], 'ro-', markersize=15, label='Red')
        ax.plot([3, 0], [0, 0], 'ko--', alpha=0.5)
        
        # Add annotations
        ax.annotate('Start', xy=(0, 0), xytext=(0, -0.2), ha='center')
        ax.annotate('Degradation', xy=(0.5, 0.5), xytext=(0.5, 0.7), ha='center')
        ax.annotate(f'Consecutive Yellow\nCount >= {self.yellow_threshold.get()}', xy=(1.5, 1), xytext=(1.5, 1.2), ha='center')
        ax.annotate('Component\nFailure', xy=(2.5, 0.5), xytext=(2.5, 0.7), ha='center')
        ax.annotate('Maintenance\n(Reset All Components)', xy=(1.5, 0), xytext=(1.5, -0.2), ha='center')
        
        # Set limits and remove axes
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add title
        ax.set_title('System State Transitions and Maintenance Policy')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def determine_truncation_level(self, C, K, alpha, epsilon):
        """Determine the truncation level U such that P{S^U_Ω} ≤ ε"""
        # Based on equation (25) from the paper
        # Simplified approach: We'll use a heuristic based on component degradation
        
        max_U = 200  # Maximum reasonable truncation level
        
        # Probability of all components staying in perfect condition for n steps
        # is (alpha)^(C*n) where alpha is the probability of NOT degrading
        
        for n in range(1, max_U):
            # Probability that at least one component has degraded
            p_degradation = 1 - (alpha ** (C * n))
            
            if p_degradation >= 1 - epsilon:
                return n
        
        return max_U

    def get_possible_transitions(self, state, alpha):
        """Calculate possible transitions from a given state based on degradation model"""
        C = len(state)
        transitions = {}
        
        # For each component, it can either stay at its current level or degrade one level
        # Generate all possible combinations of these transitions
        self.generate_transitions(state, alpha, 0, [], transitions)
        
        return transitions

    def generate_transitions(self, state, alpha, index, current_state, transitions):
        """Recursive helper to generate all possible transitions"""
        if index == len(state):
            # We've determined the transition for all components
            new_state = tuple(current_state)
            
            # Calculate probability of this transition
            prob = 1.0
            for i, (old, new) in enumerate(zip(state, new_state)):
                if old == new:
                    prob *= alpha  # No degradation
                else:
                    prob *= (1 - alpha)  # Degradation
            
            transitions[new_state] = prob
            return
        
        # Current component stays the same
        current_state.append(state[index])
        self.generate_transitions(state, alpha, index+1, current_state, transitions)
        current_state.pop()
        
        # Current component degrades by 1 level if not already at max
        K = self.K.get()
        if state[index] < K:
            current_state.append(state[index] + 1)
            self.generate_transitions(state, alpha, index+1, current_state, transitions)
            current_state.pop()

    def determine_signal(self, state, K):
        """Determine the signal based on component states"""
        # Green (0): All components are at level 0
        if all(d == 0 for d in state):
            return 0
        
        # Red (2): At least one component is at level K (failed)
        if any(d >= K for d in state):
            return 2
        
        # Yellow (1): Some degradation but no failures
        return 1

    def get_possible_signals(self, n, K):
        """Determine possible signals at time step n based on equation (9) in the paper"""
        if n == 0:
            return [0]  # Only green at initial state
        elif n < K:
            return [0, 1]  # Green or yellow for n < K
        else:
            return [0, 1, 2]  # Green, yellow, or red for n ≥ K

    def get_possible_actions(self, signal):
        """Determine possible actions based on signal"""
        C = self.C.get()
        
        if signal == 2:  # Red signal (intervention required)
            return list(range(1, C+1))  # Actions 1 to C
        else:  # Green or yellow (intervention optional)
            return list(range(C+1))  # Actions 0 to C

    def run_simulation(self):
        """Run a simulation based on the current parameters and policy"""
        self.status_label.config(text="Running simulation...", foreground="blue")
        self.log_message("Starting simulation...")
        
        # Get system parameters
        C = self.C.get()
        K = self.K.get()
        alpha = self.alpha.get()  # Probability of NOT degrading
        simulation_steps = self.simulation_steps.get()
        yellow_threshold = self.yellow_threshold.get()  # Maximum acceptable yellow states
        
        # Make sure component_params is updated
        self.update_component_params()
        
        # Initialize simulation variables
        component_states = np.zeros((C, simulation_steps + 1), dtype=int)
        current_states = np.zeros(C, dtype=int)
        signal_history = np.zeros(simulation_steps + 1, dtype=int)
        signal_history[0] = 0  # Initial signal is green
        
        # Initialize yellow counter for tracking consecutive yellow signals
        yellow_counter = 0
        yellow_threshold_reached_count = 0
        
        # Reset metrics
        self.intervention_count = 0
        self.preventive_maintenance_count = 0
        self.corrective_maintenance_count = 0
        self.failure_count = 0
        self.downtime_steps = 0
        self.maintenance_events = []
        
        # Initialize cost tracking
        maintenance_costs = np.zeros(simulation_steps + 1)
        transfer_costs = np.zeros(simulation_steps + 1)
        shortage_costs = np.zeros(simulation_steps + 1)
        excess_costs = np.zeros(simulation_steps + 1)
        component_costs = np.zeros(simulation_steps + 1)
        cumulative_costs = np.zeros(simulation_steps + 1)
        
        # Update progress bar
        progress_step = 100 / simulation_steps
        
        # Main simulation loop
        for t in range(simulation_steps):
            self.progress_var.set((t + 1) * progress_step)
            self.root.update()
            
            # Store current component states
            component_states[:, t] = current_states
            
            # Current signal is determined by component states
            current_signal = self.determine_signal(current_states, K)
            signal_history[t] = current_signal
            
            # Update yellow counter based on current signal
            if current_signal == 1:  # Yellow signal
                yellow_counter += 1
            else:
                yellow_counter = 0  # Reset counter if not yellow
            
            # Determine if maintenance is needed based on:
            # 1. Red signal (component failure) - immediate maintenance
            # 2. Yellow counter exceeding threshold
            maintenance_needed = False
            maintenance_type = None
            
            if current_signal == 2:  # Red signal - immediate maintenance
                maintenance_needed = True
                maintenance_type = "corrective"
            elif yellow_counter >= yellow_threshold:  # Yellow threshold reached
                maintenance_needed = True
                maintenance_type = "preventive"
                yellow_threshold_reached_count += 1
            
            # Execute maintenance if needed
            if maintenance_needed:
                self.intervention_count += 1
                self.maintenance_events.append(t)
                
                # Count by type
                if maintenance_type == "corrective":
                    self.corrective_maintenance_count += 1
                    maintenance_costs[t] = self.c2.get()  # Corrective maintenance cost
                else:  # preventive
                    self.preventive_maintenance_count += 1
                    maintenance_costs[t] = self.c1.get()  # Preventive maintenance cost
                
                # For maintenance we take all components
                action = C
                
                # Transfer cost
                transfer_costs[t] = action * self.ct.get()
                
                # Count actual degraded components
                degraded_count = np.sum(current_states > 0)
                
                # Replacement cost
                component_costs[t] = degraded_count * self.cr.get()
                
                # Shortage or excess cost
                if action < degraded_count:
                    shortage_costs[t] = (degraded_count - action) * self.cs.get()
                else:
                    excess_costs[t] = (action - degraded_count) * self.ce.get()
                
                # Reset all components to perfect condition
                current_states.fill(0)
                
                # Reset yellow counter after maintenance
                yellow_counter = 0
                
                # Log maintenance action
                self.log_message(f"Time {t}: {maintenance_type.capitalize()} maintenance performed. Reset all components.")
            else:
                # No intervention, degrade components according to their probabilities
                for i in range(C):
                    # Component-specific degradation rate
                    comp_alpha = self.component_params[i]['p']  # This is the degradation probability
                    
                    # Only degrade if not already at maximum level
                    if current_states[i] < K and random.random() < comp_alpha:
                        current_states[i] += 1
                
                # Check for failures
                if np.any(current_states >= K):
                    self.failure_count += 1
                    self.downtime_steps += 1
            
            # Calculate cumulative costs
            if t > 0:
                cumulative_costs[t] = (cumulative_costs[t-1] + maintenance_costs[t] + 
                                      transfer_costs[t] + component_costs[t] + 
                                      shortage_costs[t] + excess_costs[t])
            else:
                cumulative_costs[t] = (maintenance_costs[t] + transfer_costs[t] + 
                                      component_costs[t] + shortage_costs[t] + excess_costs[t])
        
        # Store final state
        component_states[:, simulation_steps] = current_states
        
        # Store simulation results
        self.simulation_results = {
            'component_states': component_states,
            'signal_history': signal_history,
            'maintenance_events': self.maintenance_events,
            'yellow_threshold_reached_count': yellow_threshold_reached_count,
            'costs': {
                'maintenance': maintenance_costs,
                'transfer': transfer_costs,
                'component': component_costs,
                'shortage': shortage_costs,
                'excess': excess_costs,
                'cumulative': cumulative_costs
            }
        }
        
        # Calculate performance metrics
        self.calculate_performance_metrics(simulation_steps)
        
        # Update status
        self.status_label.config(text="Simulation complete", foreground="green")
        self.log_message("Simulation completed successfully")
        
        # Generate visualizations
        self.create_visualizations()

    def determine_signal(self, state, K):
        """Determine the signal based on component states"""
        # Green (0): All components are at level 0
        if all(d == 0 for d in state):
            return 0
        
        # Red (2): At least one component is at level K (failed)
        if any(d >= K for d in state):
            return 2
        
        # Yellow (1): Some degradation but no failures
        return 1

    def calculate_performance_metrics(self, simulation_steps):
        """Calculate performance metrics based on simulation results"""
        # Extract necessary data
        maintenance_events = self.maintenance_events
        costs = self.simulation_results['costs']
        yellow_threshold_reached_count = self.simulation_results['yellow_threshold_reached_count']
        
        # Uptime percentage
        uptime_percentage = ((simulation_steps - self.downtime_steps) / simulation_steps) * 100
        
        # Mean time between failures (MTBF)
        if self.failure_count > 0:
            mtbf = simulation_steps / self.failure_count
        else:
            mtbf = simulation_steps  # No failures
        
        # Total cost
        total_cost = costs['cumulative'][-1]
        
        # Update GUI with metrics
        self.total_cost_label.config(text=f"{total_cost:.2f}")
        self.uptime_percentage_label.config(text=f"{uptime_percentage:.2f}%")
        self.mean_time_between_failures_label.config(text=f"{mtbf:.2f} steps")
        self.number_of_interventions_label.config(text=f"{self.intervention_count}")
        self.preventive_maintenance_count_label.config(text=f"{self.preventive_maintenance_count}")
        self.corrective_maintenance_count_label.config(text=f"{self.corrective_maintenance_count}")
        self.yellow_signal_threshold_reached_label.config(text=f"{yellow_threshold_reached_count}")
        
        # Log detailed metrics
        self.log_message(f"Total cost: {total_cost:.2f}")
        self.log_message(f"Uptime percentage: {uptime_percentage:.2f}%")
        self.log_message(f"Mean time between failures: {mtbf:.2f} steps")
        self.log_message(f"Total interventions: {self.intervention_count}")
        self.log_message(f"Preventive maintenance count: {self.preventive_maintenance_count}")
        self.log_message(f"Corrective maintenance count: {self.corrective_maintenance_count}")
        self.log_message(f"Yellow threshold reached count: {yellow_threshold_reached_count}")
        self.log_message(f"Failure count: {self.failure_count}")

    def create_visualizations(self):
        """Create all visualizations based on simulation results"""
        if not self.simulation_results:
            messagebox.showwarning("Visualization Error", "No simulation results to visualize")
            return
        
        try:
            self.create_component_state_visualization()
            self.create_heatmap_visualization()
            self.create_cost_analysis_visualization()
            self.create_signal_history_visualization()
        except Exception as e:
            self.log_message(f"Error creating visualizations: {str(e)}")
            messagebox.showerror("Visualization Error", f"Error creating visualizations: {str(e)}")

    def create_component_state_visualization(self):
        """Create visualization of signal history only"""
        # Clear existing content
        for widget in self.time_series_tab.winfo_children():
            widget.destroy()
        
        # Extract data
        signal_history = self.simulation_results['signal_history']
        maintenance_events = self.maintenance_events
        simulation_steps = self.simulation_steps.get()
        
        # Create figure
        fig = mpl_fig.Figure(figsize=(10, 6), dpi=100)
        
        # Create a single subplot for signal history
        ax = fig.add_subplot(111)
        
        # Create a legend handler manually
        legend_elements = []
        
        # Plot signal history
        for signal, color, label in [(0, 'green', 'Green'), (1, 'gold', 'Yellow'), (2, 'red', 'Red')]:
            mask = signal_history == signal
            if np.any(mask):
                points = ax.scatter(np.where(mask)[0], [signal] * np.sum(mask), 
                                   color=color, label=f"{label} ({signal})")
                legend_elements.append(points)
        
        # Add maintenance event line to legend if there are any events
        if maintenance_events:
            line = ax.axvline(x=maintenance_events[0], color='blue', linestyle='-', alpha=0.5, 
                             label='Maintenance')
            legend_elements.append(line)
            
            # Add the rest of maintenance events without adding to legend
            for event in maintenance_events[1:]:
                ax.axvline(x=event, color='blue', linestyle='-', alpha=0.5)
        
        # Set limits and labels
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Green (0)', 'Yellow (1)', 'Red (2)'])
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Signal", fontsize=12)
        ax.set_title("System Signal Over Time", fontsize=14)
        
        # Add legend manually
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add yellow threshold annotation
        yellow_threshold = self.yellow_threshold.get()
        ax.text(0.02, 0.02, f"Yellow Threshold: {yellow_threshold}", transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, master=self.time_series_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_heatmap_visualization(self):
        """Create heatmap visualization of component degradation"""
        # Clear existing content
        for widget in self.component_heatmap_tab.winfo_children():
            widget.destroy()
        
        # Extract data
        component_states = self.simulation_results['component_states']
        maintenance_events = self.maintenance_events
        C = self.C.get()
        K = self.K.get()
        simulation_steps = self.simulation_steps.get()
        
        # Create figure
        fig = mpl_fig.Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # If too many time steps, sample at regular intervals
        if simulation_steps > 50:
            sample_points = np.linspace(0, simulation_steps, 50, dtype=int)
            sampled_states = component_states[:, sample_points]
            time_labels = sample_points
        else:
            sampled_states = component_states
            time_labels = range(simulation_steps + 1)
        
        # Normalize states by component-specific thresholds
        normalized_states = np.zeros_like(sampled_states, dtype=float)
        for i in range(C):
            comp_k = self.component_params[i]['k']
            normalized_states[i, :] = sampled_states[i, :] / comp_k
        
        # Create heatmap
        im = ax.imshow(normalized_states, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Degradation Level (% of threshold)')
        
        # Set labels and ticks
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Component')
        ax.set_title('Component Degradation Heatmap')
        
        # Set y-ticks (components)
        ax.set_yticks(range(C))
        ax.set_yticklabels([self.component_params[i]['name'] for i in range(C)])
        
        # Set x-ticks (time steps) - show fewer ticks if many steps
        if len(time_labels) > 20:
            tick_indices = np.linspace(0, len(time_labels)-1, 20, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([time_labels[i] for i in tick_indices])
        else:
            ax.set_xticks(range(len(time_labels)))
            ax.set_xticklabels(time_labels)
        
        # Mark maintenance events
        for event in maintenance_events:
            if simulation_steps > 50:
                # Find nearest sample point
                event_idx = np.abs(sample_points - event).argmin()
                ax.axvline(x=event_idx, color='blue', linestyle='--', alpha=0.7)
            else:
                ax.axvline(x=event, color='blue', linestyle='--', alpha=0.7)
        
        # Add annotation for maintenance events
        if maintenance_events:
            ax.text(0.98, 0.02, 'Blue lines: Maintenance events', transform=ax.transAxes, 
                   color='blue', fontsize=10, ha='right', va='bottom', 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, master=self.component_heatmap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_cost_analysis_visualization(self):
        """Create cost analysis visualization"""
        # Clear existing content
        for widget in self.cost_analysis_tab.winfo_children():
            widget.destroy()
        
        # Extract cost data
        costs = self.simulation_results['costs']
        maintenance_events = self.maintenance_events
        simulation_steps = self.simulation_steps.get()
        
        # Create figure
        fig = mpl_fig.Figure(figsize=(12, 10), dpi=100)
        
        # 1. Stacked bar chart for cost breakdown
        ax1 = fig.add_subplot(211)
        
        # Create time points
        time_points = np.arange(simulation_steps + 1)
        
        # Plot cost components
        ax1.bar(time_points, costs['maintenance'], label='Fixed Maintenance')
        ax1.bar(time_points, costs['transfer'], bottom=costs['maintenance'], label='Transfer')
        ax1.bar(time_points, costs['component'], 
                bottom=costs['maintenance'] + costs['transfer'], label='Component Replacement')
        ax1.bar(time_points, costs['shortage'], 
                bottom=costs['maintenance'] + costs['transfer'] + costs['component'], label='Shortage')
        ax1.bar(time_points, costs['excess'], 
                bottom=costs['maintenance'] + costs['transfer'] + costs['component'] + costs['shortage'], 
                label='Excess')
        
        # Mark maintenance events
        for event in maintenance_events:
            ax1.axvline(x=event, color='black', linestyle='--', alpha=0.3)
        
        # Set labels and title
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Breakdown by Type')
        ax1.legend()
        
        # 2. Cumulative cost plot
        ax2 = fig.add_subplot(212)
        
        # Plot cumulative cost
        ax2.plot(time_points, costs['cumulative'], 'b-', linewidth=2)
        
        # Mark maintenance events
        for event in maintenance_events:
            ax2.axvline(x=event, color='r', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Cost')
        ax2.set_title('Cumulative Cost Over Time')
        
        # Add cost summary text
        total_costs = {
            'Maintenance': np.sum(costs['maintenance']),
            'Transfer': np.sum(costs['transfer']),
            'Component': np.sum(costs['component']),
            'Shortage': np.sum(costs['shortage']),
            'Excess': np.sum(costs['excess']),
            'Total': costs['cumulative'][-1]
        }
        
        summary_text = '\n'.join([f'{k}: {v:.2f}' for k, v in total_costs.items()])
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, master=self.cost_analysis_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_signal_history_visualization(self):
        """Create signal history visualization with only the pie chart"""
        # Clear existing content
        for widget in self.signal_history_tab.winfo_children():
            widget.destroy()
        
        # Extract data
        signal_history = self.simulation_results['signal_history']
        
        # Create figure
        fig = mpl_fig.Figure(figsize=(10, 8), dpi=100)
        
        # Create a single subplot for the pie chart
        ax = fig.add_subplot(111)
        
        # Count occurrences of each signal
        signal_counts = {
            'Green (0)': np.sum(signal_history == 0),
            'Yellow (1)': np.sum(signal_history == 1),
            'Red (2)': np.sum(signal_history == 2)
        }
        
        # Create labels and percentages
        labels = list(signal_counts.keys())
        sizes = list(signal_counts.values())
        
        # Calculate percentages
        total = sum(sizes)
        percentages = [100 * s / total for s in sizes]
        
        # Create customized labels to avoid overlapping
        # Just use percentages without text labels on the pie
        # The labels will be in the legend instead
        autopct_labels = ['%1.1f%%' % p for p in percentages]
        
        # Create pie chart
        colors = ['green', 'gold', 'red']
        wedges, texts, autotexts = ax.pie(sizes, colors=colors, 
                                         autopct='%1.1f%%',
                                         textprops={'fontsize': 12},
                                         startangle=90)
        
        # Customize text appearance to avoid overlapping
        for text in autotexts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
            
        # Create a legend outside the pie chart
        ax.legend(wedges, [f'{l} ({p:.1f}%)' for l, p in zip(labels, percentages)],
                loc='upper right', bbox_to_anchor=(1.0, 0.9))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add title
        ax.set_title('Signal Distribution', fontsize=16, pad=20)
        
        # Remove border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add extra space around the pie chart
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, master=self.signal_history_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Clear results
        self.simulation_results = None
        self.optimal_policy = None
        self.maintenance_events = []
        
        # Reset metrics
        self.total_cost_label.config(text="N/A")
        self.uptime_percentage_label.config(text="N/A")
        self.mean_time_between_failures_label.config(text="N/A")
        self.number_of_interventions_label.config(text="N/A")
        self.preventive_maintenance_count_label.config(text="N/A")
        self.corrective_maintenance_count_label.config(text="N/A")
        self.yellow_signal_threshold_reached_label.config(text="N/A")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Reset progress
        self.progress_var.set(0)
        
        # Update status
        self.status_label.config(text="Ready", foreground="blue")
        self.log_message("Simulation reset")
        
        # Clear visualizations
        for tab in [self.time_series_tab, self.component_heatmap_tab, 
                   self.cost_analysis_tab, self.signal_history_tab]:
            for widget in tab.winfo_children():
                widget.destroy()
        
        # Clear policy tab
        for widget in self.policy_tab.winfo_children():
            widget.destroy()
        
        ttk.Label(self.policy_tab, text="No optimal policy has been calculated yet.").pack(padx=10, pady=10)


def main():
    root = tk.Tk()
    app = MaintenanceOptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
