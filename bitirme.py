import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig


class DegradationSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Degradation Simulation")
        self.root.geometry("1200x800")

        # Initialize parameters
        self.C = tk.IntVar(value=3)  # Number of components
        self.K = tk.IntVar(value=7)  # Default failure threshold
        self.P = tk.DoubleVar(value=0.15)  # Default degradation probability
        self.simulation_steps = tk.IntVar(value=100)

        # Component-specific parameters
        self.component_params = []  # Will hold dictionaries of parameters for each component

        # Cost parameters
        self.maintenance_cost = tk.DoubleVar(value=1000)
        self.failure_cost = tk.DoubleVar(value=5000)
        self.inspection_cost = tk.DoubleVar(value=100)

        # Additional performance metrics
        self.total_cost = 0
        self.uptime_percentage = 0
        self.maintenance_efficiency = 0
        self.false_alarm_rate = 0
        self.mean_time_between_failures = 0

        # Create frames
        self.create_parameter_frame()
        self.create_cost_frame()
        self.create_button_frame()
        self.create_result_frame()
        self.create_performance_frame()
        self.create_graph_frame()

        # Simulation results
        self.component_states = None
        self.sensor_signals = None
        self.system_health = {}
        self.intervention_count = 0
        self.failure_count = 0
        self.false_alarm_count = 0
        self.downtime_steps = 0
        self.maintenance_events = []

    def create_parameter_frame(self):
        param_frame = ttk.LabelFrame(self.root, text="Simulation Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # C parameter
        ttk.Label(param_frame, text="Number of Components (C):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        c_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.C, width=5)
        c_spinbox.grid(row=0, column=1, padx=5, pady=5)
        c_spinbox.bind("<Return>", self.update_component_params)
        c_spinbox.bind("<<Increment>>", self.update_component_params)
        c_spinbox.bind("<<Decrement>>", self.update_component_params)

        # Default K parameter
        ttk.Label(param_frame, text="Default Failure Threshold (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(param_frame, from_=1, to=20, textvariable=self.K, width=5).grid(row=1, column=1, padx=5, pady=5)

        # Default P parameter
        ttk.Label(param_frame, text="Default Degradation Probability (P):").grid(row=2, column=0, padx=5, pady=5,
                                                                                 sticky="w")
        ttk.Spinbox(param_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.P, width=5).grid(row=2, column=1,
                                                                                                        padx=5, pady=5)

        # Simulation steps
        ttk.Label(param_frame, text="Simulation Steps:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(param_frame, from_=10, to=1000, increment=10, textvariable=self.simulation_steps, width=5).grid(
            row=3, column=1, padx=5, pady=5)

        # Button to edit component parameters
        ttk.Button(param_frame, text="Edit Component Parameters", command=self.open_component_editor).grid(
            row=4, column=0, columnspan=2, padx=5, pady=10)

        # Initialize component parameters
        self.update_component_params()

    def create_cost_frame(self):
        cost_frame = ttk.LabelFrame(self.root, text="Cost Parameters")
        cost_frame.grid(row=0, column=0, padx=10, pady=10, sticky="sw")
        cost_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nw")

        # Maintenance cost
        ttk.Label(cost_frame, text="Maintenance Cost:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=100, to=10000, increment=100, textvariable=self.maintenance_cost, width=7).grid(
            row=0, column=1, padx=5, pady=5)

        # Failure cost
        ttk.Label(cost_frame, text="Failure Cost:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=500, to=50000, increment=500, textvariable=self.failure_cost, width=7).grid(
            row=1, column=1, padx=5, pady=5)

        # Inspection cost
        ttk.Label(cost_frame, text="Inspection Cost:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(cost_frame, from_=10, to=1000, increment=10, textvariable=self.inspection_cost, width=7).grid(
            row=2, column=1, padx=5, pady=5)

    def create_button_frame(self):
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nw")

        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(padx=5, pady=5)

    def create_result_frame(self):
        result_frame = ttk.LabelFrame(self.root, text="Simulation Results")
        result_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nw")

        self.result_label = ttk.Label(result_frame, text="No simulation results yet.")
        self.result_label.pack(padx=5, pady=5)

    def create_performance_frame(self):
        performance_frame = ttk.LabelFrame(self.root, text="Performance Metrics")
        performance_frame.grid(row=4, column=0, padx=10, pady=5, sticky="nw")

        # Create labels for each performance metric
        self.total_cost_label = ttk.Label(performance_frame, text="Total Cost: N/A")
        self.total_cost_label.pack(anchor="w", padx=5, pady=2)

        self.uptime_label = ttk.Label(performance_frame, text="Uptime Percentage: N/A")
        self.uptime_label.pack(anchor="w", padx=5, pady=2)

        self.mtbf_label = ttk.Label(performance_frame, text="Mean Time Between Failures: N/A")
        self.mtbf_label.pack(anchor="w", padx=5, pady=2)

        self.maintenance_efficiency_label = ttk.Label(performance_frame, text="Maintenance Efficiency: N/A")
        self.maintenance_efficiency_label.pack(anchor="w", padx=5, pady=2)

        self.false_alarm_label = ttk.Label(performance_frame, text="False Alarm Rate: N/A")
        self.false_alarm_label.pack(anchor="w", padx=5, pady=2)

    def create_graph_frame(self):
        self.graph_frame = ttk.LabelFrame(self.root, text="Visualization")
        self.graph_frame.grid(row=0, column=1, rowspan=5, padx=10, pady=10, sticky="nsew")

        # Configure grid to expand graph frame
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)

        # Create notebook (tabbed interface) for the visualizations
        self.notebook = ttk.Notebook(self.graph_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs for each visualization
        self.time_series_tab = ttk.Frame(self.notebook)
        self.heatmap_tab = ttk.Frame(self.notebook)
        self.cost_analysis_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.time_series_tab, text="Time Series")
        self.notebook.add(self.heatmap_tab, text="Component Heatmap")
        self.notebook.add(self.cost_analysis_tab, text="Cost Analysis")

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
                    'name': f"Component {i + 1}",
                    'k': self.K.get(),  # Failure threshold
                    'p': self.P.get(),  # Degradation probability
                    'cost': 100.0  # Component-specific cost
                })

    def open_component_editor(self):
        """Open a window to edit component parameters"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Component Parameters Editor")
        editor_window.geometry("600x400")

        # Create a frame with scrollbar
        main_frame = ttk.Frame(editor_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add a canvas
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
        ttk.Label(scrollable_frame, text="Component Name", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5,
                                                                                            pady=5)
        ttk.Label(scrollable_frame, text="Failure Threshold (K)", font=("Arial", 10, "bold")).grid(row=0, column=1,
                                                                                                   padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Degradation Prob (P)", font=("Arial", 10, "bold")).grid(row=0, column=2,
                                                                                                  padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Component Cost", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5,
                                                                                            pady=5)

        # Component parameter fields
        name_vars = []
        k_vars = []
        p_vars = []
        cost_vars = []

        for i, comp in enumerate(self.component_params):
            # Variables to hold values
            name_var = tk.StringVar(value=comp['name'])
            k_var = tk.IntVar(value=comp['k'])
            p_var = tk.DoubleVar(value=comp['p'])
            cost_var = tk.DoubleVar(value=comp['cost'])

            name_vars.append(name_var)
            k_vars.append(k_var)
            p_vars.append(p_var)
            cost_vars.append(cost_var)

            # Create entry fields
            ttk.Entry(scrollable_frame, textvariable=name_var).grid(row=i + 1, column=0, padx=5, pady=2)
            ttk.Spinbox(scrollable_frame, from_=1, to=20, textvariable=k_var, width=5).grid(row=i + 1, column=1, padx=5,
                                                                                            pady=2)
            ttk.Spinbox(scrollable_frame, from_=0.01, to=1.0, increment=0.01, textvariable=p_var, width=5).grid(
                row=i + 1, column=2, padx=5, pady=2)
            ttk.Spinbox(scrollable_frame, from_=1, to=10000, increment=10, textvariable=cost_var, width=8).grid(
                row=i + 1, column=3, padx=5, pady=2)

        # Save button
        def save_parameters():
            for i in range(len(self.component_params)):
                self.component_params[i]['name'] = name_vars[i].get()
                self.component_params[i]['k'] = k_vars[i].get()
                self.component_params[i]['p'] = p_vars[i].get()
                self.component_params[i]['cost'] = cost_vars[i].get()
            editor_window.destroy()

        ttk.Button(editor_window, text="Save Changes", command=save_parameters).pack(pady=10)

    def degrade_components(self, states, P=None):
        """ Degrade components based on a Bernoulli distribution with component-specific probabilities. """
        for i in range(len(states)):
            # Get component-specific failure threshold and degradation probability
            comp_k = self.component_params[i]['k']
            comp_p = self.component_params[i]['p'] if P is None else P

            if states[i] < comp_k:  # Use component-specific threshold
                if random.random() < comp_p:
                    states[i] += 1
        return states

    def observe_system(self, states):
        """ Aggregate sensor signal σ based on component-specific thresholds. """
        # Check if any component has failed based on its specific threshold
        if any(states[i] >= self.component_params[i]['k'] for i in range(len(states))):
            return 2  # Red: at least one component has failed
        elif any(state > 0 for state in states):
            return 1  # Yellow: at least one component is degraded, but none are failed
        else:
            return 0  # Green: all components are operational

    def perform_maintenance(self, states):
        """ Perform maintenance and reset all components to state 0. """
        states.fill(0)
        return states

    def calculate_costs(self, simulation_steps):
        """Calculate total costs based on maintenance events, failures, and inspections."""
        # Calculate component-specific costs
        component_replacement_costs = sum(comp['cost'] for comp in self.component_params) * self.intervention_count

        # Add maintenance labor cost
        maintenance_labor_cost = self.maintenance_cost.get() * self.intervention_count

        # Total maintenance cost combines parts and labor
        maintenance_cost = component_replacement_costs + maintenance_labor_cost

        # Other costs
        failure_cost = self.failure_cost.get() * self.failure_count
        inspection_cost = self.inspection_cost.get() * simulation_steps  # Inspection done at every step

        return {
            'maintenance_cost': maintenance_cost,
            'component_replacement_cost': component_replacement_costs,
            'maintenance_labor_cost': maintenance_labor_cost,
            'failure_cost': failure_cost,
            'inspection_cost': inspection_cost,
            'total_cost': maintenance_cost + failure_cost + inspection_cost
        }

    def calculate_performance_metrics(self, simulation_steps):
        """Calculate performance metrics from the simulation."""
        # Uptime percentage (proportion of time not in failure state)
        uptime_percentage = ((simulation_steps - self.downtime_steps) / simulation_steps) * 100

        # Mean time between failures
        if self.failure_count > 0:
            mtbf = simulation_steps / self.failure_count
        else:
            mtbf = simulation_steps  # No failures occurred

        # Maintenance efficiency (cost per unit of uptime)
        total_cost = self.calculate_costs(simulation_steps)['total_cost']
        if uptime_percentage > 0:
            maintenance_efficiency = total_cost / uptime_percentage
        else:
            maintenance_efficiency = float('inf')  # Avoid division by zero

        # False alarm rate (interventions that weren't actually needed)
        if self.intervention_count > 0:
            false_alarm_rate = self.false_alarm_count / self.intervention_count * 100
        else:
            false_alarm_rate = 0

        return {
            'uptime_percentage': uptime_percentage,
            'mtbf': mtbf,
            'maintenance_efficiency': maintenance_efficiency,
            'false_alarm_rate': false_alarm_rate,
            'total_cost': total_cost
        }

    def run_simulation(self):
        # Clear any existing figures
        for widget in self.time_series_tab.winfo_children():
            widget.destroy()
        for widget in self.heatmap_tab.winfo_children():
            widget.destroy()
        for widget in self.cost_analysis_tab.winfo_children():
            widget.destroy()

        # Get parameters from the GUI
        C = self.C.get()
        simulation_steps = self.simulation_steps.get()

        # Make sure component_params is updated with the correct number of components
        self.update_component_params()

        # Initialize component states
        self.component_states = np.zeros((C, simulation_steps), dtype=int)
        current_states = np.zeros(C, dtype=int)
        self.sensor_signals = np.zeros(simulation_steps, dtype=int)

        # Reset system health, intervention count, and performance metrics
        self.system_health = {}
        self.intervention_count = 0
        self.failure_count = 0
        self.false_alarm_count = 0
        self.downtime_steps = 0
        self.maintenance_events = []

        # Initialize cost tracking
        maintenance_costs = np.zeros(simulation_steps)
        failure_costs = np.zeros(simulation_steps)
        inspection_costs = np.zeros(simulation_steps)
        cumulative_costs = np.zeros(simulation_steps)

        # Main simulation loop
        for t in range(simulation_steps):
            # Add inspection cost at every step
            inspection_costs[t] = self.inspection_cost.get()

            # Degrade components using component-specific parameters
            current_states = self.degrade_components(current_states)
            self.component_states[:, t] = current_states.copy()

            # Observe the system health
            sensor_signal = self.observe_system(current_states)
            self.sensor_signals[t] = sensor_signal

            # Track system health
            self.system_health[t] = {
                'component_states': current_states.copy(),
                'sensor_signal': sensor_signal
            }

            # Check for failures (signal == 2)
            if sensor_signal == 2:
                self.failure_count += 1
                failure_costs[t] = self.failure_cost.get()
                self.downtime_steps += 1

                # Maintenance decision (intervention when at least one component has failed)
                current_states = self.perform_maintenance(current_states)
                self.intervention_count += 1

                # Add component-specific costs for maintenance
                maintenance_costs[t] = self.maintenance_cost.get()
                for comp in self.component_params:
                    maintenance_costs[t] += comp['cost']

                self.maintenance_events.append(t)

            # Calculate cumulative costs
            if t > 0:
                cumulative_costs[t] = cumulative_costs[t - 1] + maintenance_costs[t] + failure_costs[t] + \
                                      inspection_costs[t]
            else:
                cumulative_costs[t] = maintenance_costs[t] + failure_costs[t] + inspection_costs[t]

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(simulation_steps)
        costs = self.calculate_costs(simulation_steps)

        # Update result label
        self.result_label.config(
            text=f"Total number of interventions: {self.intervention_count}, Failures: {self.failure_count}")

        # Update performance metrics labels
        self.total_cost_label.config(text=f"Total Cost: {costs['total_cost']:.2f}")
        self.uptime_label.config(text=f"Uptime Percentage: {metrics['uptime_percentage']:.2f}%")
        self.mtbf_label.config(text=f"Mean Time Between Failures: {metrics['mtbf']:.2f} steps")
        self.maintenance_efficiency_label.config(
            text=f"Maintenance Efficiency: {metrics['maintenance_efficiency']:.2f}")
        self.false_alarm_label.config(text=f"False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")

        # Create and display the visualizations
        self.plot_system_health_original()
        self.plot_heatmap_visualization()
        self.plot_cost_analysis(maintenance_costs, failure_costs, inspection_costs, cumulative_costs)

    def plot_system_health_original(self):
        """ Plot component states and sensor signals over time. """
        fig = mpl_fig.Figure(figsize=(10, 8), dpi=100)

        # Plot each component state separately
        C = self.C.get()
        simulation_steps = self.simulation_steps.get()

        for i in range(C):
            ax = fig.add_subplot(C + 1, 1, i + 1)
            comp_data = [self.system_health[t]['component_states'][i] for t in range(simulation_steps)]
            ax.plot(range(simulation_steps), comp_data,
                    label=f'{self.component_params[i]["name"]}')

            # Add component failure threshold as horizontal line
            comp_k = self.component_params[i]['k']
            ax.axhline(y=comp_k, color='r', linestyle='--',
                       label=f'Failure Threshold (K={comp_k})')

            ax.set_ylim(0, max(comp_k + 1, max(comp_data) + 1))  # Ensure Y-axis has room
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Degradation Level")
            ax.legend()
            ax.grid()
            ax.set_title(f"{self.component_params[i]['name']} State Over Time (P={self.component_params[i]['p']})")

        # Plot sensor signal as colored rectangular blocks based on state
        ax = fig.add_subplot(C + 1, 1, C + 1)

        # Color mapping for states
        colors = {0: 'green', 1: 'gold', 2: 'red'}
        labels = {0: 'Green (0)', 1: 'Yellow (1)', 2: 'Red (2)'}

        # Plot colored rectangular blocks for each time period
        for t in range(simulation_steps):
            state = self.sensor_signals[t]
            ax.add_patch(plt.Rectangle((t - 0.5, state - 0.4), 1, 0.8,
                                       color=colors[state], alpha=0.8))

        # Add legend manually
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black',
                                 label=labels[i]) for i in range(3)]
        ax.legend(handles=legend_elements, loc='best')

        ax.set_xlim(-0.5, simulation_steps - 0.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Green (0)', 'Yellow (1)', 'Red (2)'])
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sensor Signal")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title("System Health (Sensor Signal Over Time)")

        fig.tight_layout()

        # Add the figure to the tab
        canvas = FigureCanvasTkAgg(fig, master=self.time_series_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_heatmap_visualization(self):
        """Create a heatmap visualization of component degradation states over time."""
        fig = mpl_fig.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        C = self.C.get()

        # We'll sample time points to avoid overcrowding the heatmap
        simulation_steps = self.simulation_steps.get()

        # If we have too many time steps, sample a subset
        if simulation_steps > 20:
            # Sample 20 evenly spaced time points
            time_points = np.linspace(0, simulation_steps - 1, 20, dtype=int)
        else:
            time_points = np.arange(simulation_steps)

        # Create the matrix for the heatmap
        # Rows are components, columns are time points
        matrix = np.zeros((C, len(time_points)))

        # Fill the matrix with component states at selected time points
        for i, t in enumerate(time_points):
            for j in range(C):
                matrix[j, i] = self.component_states[j, t]

        # Get component-specific failure thresholds
        k_values = [self.component_params[i]['k'] for i in range(C)]

        # Calculate the normalized degradation (0-100% of component-specific K)
        normalized_matrix = np.zeros_like(matrix, dtype=float)
        for j in range(C):
            normalized_matrix[j, :] = (matrix[j, :] / k_values[j]) * 100

        # Use a green-yellow-red colormap
        cmap = plt.cm.get_cmap('RdYlGn_r')  # Red-Yellow-Green reversed

        # Plot the heatmap
        sns.heatmap(normalized_matrix, cmap=cmap, annot=matrix.astype(int), fmt="d",
                    cbar_kws={'label': 'Degradation Level (%)'}, ax=ax)

        # Set the labels
        ax.set_ylabel('Component')
        ax.set_xlabel('Time Step')

        # Set y-axis ticks (components)
        ax.set_yticks(np.arange(C) + 0.5)
        ax.set_yticklabels([self.component_params[i]['name'] for i in range(C)])

        # Set x-axis ticks (time points)
        ax.set_xticks(np.arange(len(time_points)) + 0.5)
        ax.set_xticklabels([str(t) for t in time_points], rotation=45)

        # Add title
        ax.set_title('Component Degradation Heatmap Over Time')

        # Adjust layout
        fig.tight_layout()

        # Add the figure to the tab
        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add explanation text
        explanation = ttk.Label(self.heatmap_tab,
                                text="Bu heatmap, her komponentin zaman içindeki bozulma durumunu gösterir.\nRenk skalası: Yeşil (sağlıklı) -> Sarı (orta derece bozulma) -> Kırmızı (kritik bozulma)")
        explanation.pack(pady=5)

    def plot_cost_analysis(self, maintenance_costs, failure_costs, inspection_costs, cumulative_costs):
        """Create a cost analysis visualization."""
        fig = mpl_fig.Figure(figsize=(10, 8), dpi=100)

        # First subplot: Individual costs per time step
        ax1 = fig.add_subplot(211)
        simulation_steps = self.simulation_steps.get()
        time_steps = np.arange(simulation_steps)

        # Plot individual costs
        ax1.bar(time_steps, maintenance_costs, label='Maintenance Costs')
        ax1.bar(time_steps, failure_costs, bottom=maintenance_costs, label='Failure Costs')
        ax1.bar(time_steps, inspection_costs,
                bottom=maintenance_costs + failure_costs, label='Inspection Costs')

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Breakdown per Time Step')
        ax1.legend()

        # Second subplot: Cumulative costs over time
        ax2 = fig.add_subplot(212)
        ax2.plot(time_steps, cumulative_costs, 'b-', linewidth=2)

        # Mark maintenance events on the cumulative cost curve
        for event in self.maintenance_events:
            ax2.axvline(x=event, color='r', linestyle='--', alpha=0.5)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Cost')
        ax2.set_title('Cumulative Costs Over Time (Red lines indicate maintenance events)')

        # Add cost breakdown in text format
        costs = self.calculate_costs(simulation_steps)
        cost_text = (f"Maintenance Costs: {costs['maintenance_cost']:.2f}\n"
                     f"  - Component Replacement: {costs['component_replacement_cost']:.2f}\n"
                     f"  - Maintenance Labor: {costs['maintenance_labor_cost']:.2f}\n"
                     f"Failure Costs: {costs['failure_cost']:.2f}\n"
                     f"Inspection Costs: {costs['inspection_cost']:.2f}\n"
                     f"Total Cost: {costs['total_cost']:.2f}")

        ax2.text(0.02, 0.85, cost_text, transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        fig.tight_layout()

        # Add the figure to the tab
        canvas = FigureCanvasTkAgg(fig, master=self.cost_analysis_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = DegradationSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()