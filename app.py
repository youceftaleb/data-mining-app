#!/home/youcef/.venvs/default/bin/python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk,simpledialog, Toplevel, Button
import pandas as pd
from scipy.io import arff
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import pairwise_distances
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class ClusterVisualizationWindow(tk.Toplevel):
    def __init__(self, parent, X, labels, feature_names, algorithm):
        super().__init__(parent)
        self.title(f"{algorithm} Clustering Results")
        self.geometry("1000x800")
        
        self.X = X
        self.labels = labels
        self.feature_names = feature_names
        self.algorithm = algorithm
        
        self.create_widgets()
        self.show_default_plot()
        
    def create_widgets(self):
        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="X Axis:").grid(row=0, column=0)
        self.x_axis = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.x_axis.current(0)
        self.x_axis.grid(row=0, column=1)
        
        ttk.Label(control_frame, text="Y Axis:").grid(row=0, column=2)
        self.y_axis = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.y_axis.current(1 if len(self.feature_names) > 1 else 0)
        self.y_axis.grid(row=0, column=3)
        
        ttk.Button(control_frame, text="Update Plot", command=self.update_plot).grid(row=0, column=4)
        
        # 3D checkbox if we have 3+ features
        if len(self.feature_names) >= 3:
            self.use_3d = tk.BooleanVar(value=False)
            ttk.Checkbutton(control_frame, text="3D Plot", variable=self.use_3d,
                           command=self.toggle_3d).grid(row=0, column=5)
            self.z_label = ttk.Label(control_frame, text="Z Axis:")
            self.z_axis = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
            self.z_axis.current(2)
            
        # Plot frame
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(expand=True, fill=tk.BOTH)
        
    def toggle_3d(self):
        """Show/hide 3D controls based on checkbox"""
        if self.use_3d.get():
            self.z_label.grid(row=0, column=6)
            self.z_axis.grid(row=0, column=7)
        else:
            self.z_label.grid_remove()
            self.z_axis.grid_remove()
        
    def show_default_plot(self):
        self.update_plot()
        
    def update_plot(self):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Get selected dimensions
        x_idx = self.feature_names.index(self.x_axis.get())
        y_idx = self.feature_names.index(self.y_axis.get())
        
        fig = plt.figure(figsize=(10, 8))
        
        # Check if we should do 3D plot
        if hasattr(self, 'use_3d') and self.use_3d.get() and len(self.feature_names) >= 3:
            z_idx = self.feature_names.index(self.z_axis.get())
            ax = fig.add_subplot(111, projection='3d')
            
            # Handle noise points for DBSCAN
            if self.algorithm == "DBSCAN" and -1 in self.labels:
                noise_mask = (self.labels == -1)
                ax.scatter(self.X[noise_mask, x_idx], self.X[noise_mask, y_idx], self.X[noise_mask, z_idx],
                          c='gray', alpha=0.3, s=20, label='Noise')
                cluster_mask = ~noise_mask
                scatter = ax.scatter(self.X[cluster_mask, x_idx], self.X[cluster_mask, y_idx], self.X[cluster_mask, z_idx],
                                    c=self.labels[cluster_mask], cmap='viridis', alpha=0.6, s=50)
            else:
                scatter = ax.scatter(self.X[:, x_idx], self.X[:, y_idx], self.X[:, z_idx],
                                   c=self.labels, cmap='viridis', alpha=0.6, s=50)
            
            ax.set_zlabel(self.feature_names[z_idx])
        else:
            ax = fig.add_subplot(111)
            
            # Handle noise points for DBSCAN
            if self.algorithm == "DBSCAN" and -1 in self.labels:
                noise_mask = (self.labels == -1)
                ax.scatter(self.X[noise_mask, x_idx], self.X[noise_mask, y_idx],
                          c='gray', alpha=0.3, s=20, label='Noise')
                cluster_mask = ~noise_mask
                scatter = ax.scatter(self.X[cluster_mask, x_idx], self.X[cluster_mask, y_idx],
                                    c=self.labels[cluster_mask], cmap='viridis', alpha=0.6, s=50)
            else:
                scatter = ax.scatter(self.X[:, x_idx], self.X[:, y_idx],
                                   c=self.labels, cmap='viridis', alpha=0.6, s=50)
        
        ax.set_xlabel(self.feature_names[x_idx])
        ax.set_ylabel(self.feature_names[y_idx])
        
        # Special title for AGNES
        if self.algorithm == "AGNES":
            ax.set_title(f"{self.algorithm} Clustering (Cut Tree)")
        else:
            ax.set_title(f"{self.algorithm} Clustering")
            
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.legend()
        plt.tight_layout()
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)



class DIANA:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        
    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with all points in one cluster
        clusters = [list(range(n_samples))]
        
        while len(clusters) < self.n_clusters:
            # Find the cluster with largest diameter to split
            max_diameter = -1
            cluster_to_split = None
            
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                    
                # Calculate diameter (maximum pairwise distance)
                distances = pairwise_distances(X[cluster])
                diameter = np.max(distances)
                
                if diameter > max_diameter:
                    max_diameter = diameter
                    cluster_to_split = cluster
            
            if cluster_to_split is None:
                break
                
            # Split the cluster using the farthest points
            sub_distances = pairwise_distances(X[cluster_to_split])
            max_idx = np.unravel_index(np.argmax(sub_distances), sub_distances.shape)
            p1, p2 = cluster_to_split[max_idx[0]], cluster_to_split[max_idx[1]]
            
            # Create two new clusters based on proximity to p1 or p2
            new_cluster1 = []
            new_cluster2 = []
            
            for point in cluster_to_split:
                dist_p1 = np.linalg.norm(X[point] - X[p1])
                dist_p2 = np.linalg.norm(X[point] - X[p2])
                
                if dist_p1 < dist_p2:
                    new_cluster1.append(point)
                else:
                    new_cluster2.append(point)
            
            # Update the clusters list
            clusters.remove(cluster_to_split)
            clusters.append(new_cluster1)
            clusters.append(new_cluster2)
        
        # Create labels array
        labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                labels[point] = cluster_idx
                
        self.labels_ = labels
        return labels

class KMedoids:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit_predict(self, X):
        X = np.array(X)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        medoids_idx = rng.choice(n_samples, self.n_clusters, replace=False)
        medoids = X[medoids_idx]
        
        for _ in range(self.max_iter):
            distances = pairwise_distances(X, medoids)
            labels = np.argmin(distances, axis=1)
            
            new_medoids = np.zeros_like(medoids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    cluster_distances = pairwise_distances(cluster_points)
                    total_distances = np.sum(cluster_distances, axis=1)
                    new_medoids[i] = cluster_points[np.argmin(total_distances)]
            
            if np.all(medoids == new_medoids):
                break
            medoids = new_medoids
            
        self.medoids_ = medoids
        self.labels_ = labels
        return labels

class DataMiningApp:
    def __init__(self, root):
        self.df_clean = None  # Cleaned version of the data
        self.root = root
        self.root.title("Data Mining GUI")
        self.root.geometry("1000x700")

        self.df = None

        # Upload Button
        self.upload_btn = ttk.Button(root, text="Upload CSV or ARFF", command=self.upload_file)
        self.upload_btn.pack(pady=10)

        # Treeview for displaying data
        self.tree = ttk.Treeview(root)
        self.tree.pack(expand=True, fill='both', padx=10, pady=5)

        # Stats label
        self.stats_text = tk.Text(root, height=10)
        self.stats_text.pack(fill='x', padx=10)

        # Plot section
        self.plot_frame = ttk.LabelFrame(root, text="Plotting Options")
        self.plot_frame.pack(fill='x', padx=10, pady=10)

        # Boxplot: Single attribute
        ttk.Label(self.plot_frame, text="Attribute for Boxplot:").grid(row=0, column=0, padx=5, pady=5)
        self.boxplot_attr = ttk.Combobox(self.plot_frame, state="readonly")
        self.boxplot_attr.grid(row=0, column=1, padx=5)
        ttk.Button(self.plot_frame, text="Show Boxplot", command=self.show_boxplot).grid(row=0, column=2, padx=5)

        # Scatter plot: Two attributes
        ttk.Label(self.plot_frame, text="X Axis:").grid(row=1, column=0, padx=5, pady=5)
        self.scatter_x = ttk.Combobox(self.plot_frame, state="readonly")
        self.scatter_x.grid(row=1, column=1, padx=5)

        ttk.Label(self.plot_frame, text="Y Axis:").grid(row=1, column=2, padx=5)
        self.scatter_y = ttk.Combobox(self.plot_frame, state="readonly")
        self.scatter_y.grid(row=1, column=3, padx=5)

        ttk.Button(self.plot_frame, text="Show Scatter Plot", command=self.show_scatter).grid(row=1, column=4, padx=5)
        ttk.Button(self.plot_frame, text="Handle Missing Values", command=self.open_missing_window).grid(row=2, column=0, columnspan=5, pady=10)

        # Normalization Frame
        self.norm_frame = ttk.LabelFrame(root, text="Normalization Options")
        self.norm_frame.pack(fill='x', padx=10, pady=10)

        # Column selection
        ttk.Label(self.norm_frame, text="Column to Normalize:").grid(row=0, column=0, padx=5, pady=5)
        self.norm_columns = ttk.Combobox(self.norm_frame, state="readonly")
        self.norm_columns.grid(row=0, column=1, padx=5)
        
        # Normalization Strategy selection
        ttk.Label(self.norm_frame, text="Normalization Strategy:").grid(row=1, column=0, padx=5, pady=5)
        self.norm_strategy = ttk.Combobox(self.norm_frame, values=["Min-Max Normalization", "Z-Score Normalization"], state="readonly")
        self.norm_strategy.grid(row=1, column=1, padx=5)

        # Min-Max Custom range entry (hidden by default)
        ttk.Label(self.norm_frame, text="Min value:").grid(row=2, column=0, padx=5, pady=5)
        self.min_entry = ttk.Entry(self.norm_frame)
        self.min_entry.grid(row=2, column=1, padx=5, pady=5)
        self.min_entry.configure(state='disabled')

        ttk.Label(self.norm_frame, text="Max value:").grid(row=3, column=0, padx=5, pady=5)
        self.max_entry = ttk.Entry(self.norm_frame)
        self.max_entry.grid(row=3, column=1, padx=5, pady=5)
        self.max_entry.configure(state='disabled')

        # Button to apply normalization
        ttk.Button(self.norm_frame, text="Apply Normalization", command=self.apply_normalization).grid(row=4, column=0, columnspan=2, pady=5)

        # Handle strategy change to enable/disable min/max entry fields
        self.norm_strategy.bind("<<ComboboxSelected>>", self.strategy_changed)

        # Add histogram button
        ttk.Button(self.plot_frame, text="Show Histogram", command=self.show_histogram).grid(row=0, column=3, padx=5)

        # Add export button
        self.export_btn = ttk.Button(root, text="Export Data", command=self.export_data)
        self.export_btn.pack(pady=5)

        # Add clustering section
        self.cluster_frame = ttk.LabelFrame(root, text="Clustering Algorithms")
        self.cluster_frame.pack(fill='x', padx=10, pady=10)
        # Add Dendrogram button
        ttk.Button(self.cluster_frame, text="AGNES (Dendrogram)", command=self.run_agnes).grid(row=0, column=5, padx=5)
        
        # Algorithm selection
        ttk.Label(self.cluster_frame, text="Algorithm:").grid(row=0, column=0)
        self.cluster_algo = ttk.Combobox(self.cluster_frame, 
                                       values=["K-Means", "K-Medoids", "DBSCAN", "DIANA"],
                                       state="readonly")
        self.cluster_algo.grid(row=0, column=1)
        
        # Number of clusters
        ttk.Label(self.cluster_frame, text="Number of clusters:").grid(row=0, column=2)
        self.n_clusters = ttk.Spinbox(self.cluster_frame, from_=2, to=10, width=5)
        self.n_clusters.grid(row=0, column=3)
        
        # Run button
        ttk.Button(self.cluster_frame, 
                  text="Run Clustering",
                  command=self.run_clustering).grid(row=0, column=4, padx=5)
        
        # Elbow Method Button (keep this from previous implementation)
        self.elbow_btn = ttk.Button(root, 
                                  text="Find Optimal K (Elbow Method)", 
                                  command=self.run_elbow_method)
        self.elbow_btn.pack(pady=10)

        # DBSCAN parameters (initially hidden)
        self.dbscan_frame = ttk.Frame(self.cluster_frame)
        self.dbscan_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)

        ttk.Label(self.dbscan_frame, text="EPS:").grid(row=0, column=0)
        self.dbscan_eps = ttk.Entry(self.dbscan_frame, width=8)
        self.dbscan_eps.grid(row=0, column=1)
        self.dbscan_eps.insert(0, "0.5")

        ttk.Label(self.dbscan_frame, text="Min Samples:").grid(row=0, column=2)
        self.dbscan_min_samples = ttk.Spinbox(self.dbscan_frame, from_=1, to=100, width=5)
        self.dbscan_min_samples.grid(row=0, column=3)
        self.dbscan_min_samples.set(5)

        self.dbscan_frame.grid_remove()  # Hide initially
        self.cluster_algo.bind("<<ComboboxSelected>>", self.on_algorithm_change)


    def run_agnes(self):
        """Run AGNES hierarchical clustering and show dendrogram"""
        if self.df_clean is None:
            messagebox.showerror("Error", "No data loaded")
            return

        try:
            # Get numeric data
            numeric_data = self.df_clean.select_dtypes(include=['number'])
            if len(numeric_data.columns) < 2:
                messagebox.showerror("Error", "Need at least 2 numeric columns")
                return

            X = StandardScaler().fit_transform(numeric_data)
            feature_names = numeric_data.columns.tolist()
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(X, method='ward')
            
            # Create dendrogram window with cut option
            self.show_agnes_dendrogram(linkage_matrix, X, feature_names)

        except Exception as e:
            messagebox.showerror("Error", f"AGNES clustering failed:\n{str(e)}")
    def show_agnes_dendrogram(self, linkage_matrix, X, feature_names):
        """Show AGNES dendrogram with interactive cut option"""
        dendro_window = Toplevel(self.root)
        dendro_window.title("AGNES Dendrogram")
        dendro_window.geometry("1000x800")
        
        fig = plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title('AGNES Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index or Cluster Size')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        # Add cut button
        cut_button = Button(dendro_window, text="Cut Tree and Show Clusters",
                        command=lambda: self.cut_agnes_tree(
                            linkage_matrix, X, feature_names, dendro_window))
        cut_button.pack(pady=10)
        
        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=dendro_window)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

    def cut_agnes_tree(self, linkage_matrix, X, feature_names, parent_window):
        """Cut the dendrogram at selected height and show clusters"""
        try:
            # Ask for cut height
            height = simpledialog.askfloat(
                "Cut Dendrogram",
                "Enter cut height (or leave blank to specify clusters):",
                parent=parent_window,
                minvalue=0
            )
            
            if height is None:  # User clicked cancel
                return
                
            # Get number of clusters from cut
            if height:
                clusters = cut_tree(linkage_matrix, height=height).flatten()
            else:
                n_clusters = simpledialog.askinteger(
                    "Number of Clusters",
                    "Enter number of clusters:",
                    parent=parent_window,
                    minvalue=1
                )
                if n_clusters is None:
                    return
                clusters = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()
            
            # Show cluster visualization
            ClusterVisualizationWindow(self.root, X, clusters, feature_names, "AGNES")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cut tree:\n{str(e)}")
    def on_algorithm_change(self, event):
        algo = self.cluster_algo.get()
        if algo == "DBSCAN":
            self.n_clusters.grid_remove()
            self.dbscan_frame.grid()
        else:
            self.dbscan_frame.grid_remove()
            self.n_clusters.grid()

    def plot_dendrogram(self):
        if self.df_clean is None:
            messagebox.showerror("Error", "No data loaded")
            return

        # Select numeric data
        numeric_data = self.df_clean.select_dtypes(include=['number'])
        if len(numeric_data.columns) < 2:
            messagebox.showerror("Error", "Need at least 2 numeric columns")
            return

        try:
            # Standardize the data
            X = StandardScaler().fit_transform(numeric_data)

            # Compute the linkage matrix
            linked = linkage(X, method='ward')  # 'ward' is good for numerical data

            # Plot the dendrogram
            plt.figure(figsize=(10, 6))
            dendrogram(linked, truncate_mode='level', p=5)  # show only top levels to keep it readable
            plt.title('Dendrogram (AGNES)')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot dendrogram: {str(e)}")


    def run_elbow_method(self):
        """Simple elbow method implementation"""
        if self.df_clean is None:
            messagebox.showerror("Error", "No data loaded")
            return

        # Get only numeric columns
        numeric_data = self.df_clean.select_dtypes(include=['number'])
        if len(numeric_data.columns) < 2:
            messagebox.showerror("Error", "Need at least 2 numeric columns")
            return

        try:
            # Prepare data
            X = StandardScaler().fit_transform(numeric_data)
            
            # Calculate inertias for k=1 to k=10
            inertias = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=16, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            # Plot
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, 11), inertias, 'bo-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.xticks(range(1, 11))
            plt.grid(True)
            plt.show()

            messagebox.showinfo("Done", "Elbow plot generated.\nLook for the 'bend' to choose k.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")




    def show_histogram(self):
        col = self.boxplot_attr.get()
        if col and col in self.df_clean.columns:
            plt.figure(figsize=(6, 4))
            self.df_clean[col].hist(bins=20)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def export_data(self):
        if self.df_clean is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
        )
        if not file_path:
            return
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                self.df_clean.to_csv(file_path, index=False)
            elif ext == '.xlsx':
                self.df_clean.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Data exported to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("ARFF Files", "*.arff")])
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1]
            if ext == '.csv':
                self.df = pd.read_csv(file_path)
            elif ext == '.arff':
                data, meta = arff.loadarff(file_path)
                # Convert byte strings to regular strings
                str_data = {}
                for col in data.dtype.names:
                    if data[col].dtype.kind == 'S':  # If byte string
                        str_data[col] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in data[col]]
                    else:
                        str_data[col] = data[col]
                self.df = pd.DataFrame(str_data)
                self.df_clean = self.df.copy()
            else:
                raise ValueError("Unsupported file type")
            
            self.df_clean = self.df.copy()
            self.show_data()
            self.show_stats()
            self.populate_comboboxes()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_data(self):
        # Clear old data
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.df_clean.columns)
        self.tree["show"] = "headings"

        for col in self.df_clean.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in self.df_clean.head(20).iterrows():
            self.tree.insert("", "end", values=list(row))

    def show_stats(self):
        stats = self.df_clean.describe(include='all').to_string()
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)

    def populate_comboboxes(self):
        # Only numerical columns will be considered for normalization
        numeric_columns = list(self.df_clean.select_dtypes(include=['number']).columns)  
        self.boxplot_attr['values'] = numeric_columns
        self.scatter_x['values'] = numeric_columns
        self.scatter_y['values'] = numeric_columns
        self.norm_columns['values'] = numeric_columns  # Populate normalization columns combobox with numeric columns
        if numeric_columns:
            self.boxplot_attr.current(0)
            self.scatter_x.current(0)
            self.scatter_y.current(min(1, len(numeric_columns) - 1))

    def show_boxplot(self):
        col = self.boxplot_attr.get()
        if col and col in self.df_clean.columns:
            plt.figure(figsize=(6, 4))
            self.df_clean.boxplot(column=col)
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Error", "Invalid column selected for boxplot.")

    def show_scatter(self):
        x_col = self.scatter_x.get()
        y_col = self.scatter_y.get()
        if x_col and y_col and x_col in self.df_clean.columns and y_col in self.df_clean.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(self.df_clean[x_col], self.df_clean[y_col], alpha=0.6)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Scatter Plot: {x_col} vs {y_col}")
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Error", "Invalid columns selected for scatter plot.")
    
    def open_missing_window(self):
        if self.df_clean is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        missing_cols = self.df_clean.columns[self.df_clean.isnull().any()].tolist()

        if not missing_cols:
            messagebox.showinfo("No Missing Values", "There are no missing values in the dataset.")
            return

        missing_window = tk.Toplevel(self.root)
        missing_window.title("Handle Missing Values")
        missing_window.geometry("500x300")

        ttk.Label(missing_window, text="Missing Values per Column:").pack(pady=5)

        # Show missing value count
        missing_text = tk.Text(missing_window, height=5)
        missing_text.pack(padx=10, fill='x')
        missing_counts = self.df_clean.isnull().sum()
        missing_text.insert(tk.END, str(missing_counts[missing_counts > 0]))

        # Column selection
        numeric_cols = list(self.df_clean.select_dtypes(include='number').columns)
        ttk.Label(missing_window, text="Select Column:").pack(pady=5)
        col_box = ttk.Combobox(missing_window, values=numeric_cols, state="readonly")
        col_box.pack()

        # Strategy selection
        ttk.Label(missing_window, text="Select Strategy:").pack(pady=5)
        strategy_box = ttk.Combobox(missing_window, values=["Drop Rows", "Fill with Min", "Fill with Max", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"], state="readonly")
        strategy_box.pack()

        # Custom value entry
        custom_value = tk.StringVar()
        custom_entry = ttk.Entry(missing_window, textvariable=custom_value)
        custom_entry.pack(pady=5)
        custom_entry.configure(state='disabled')

        # Enable custom entry if selected
        def strategy_changed(event):
            if strategy_box.get() == "Fill with Custom Value":
                custom_entry.configure(state='normal')
            else:
                custom_entry.configure(state='disabled')
        strategy_box.bind("<<ComboboxSelected>>", strategy_changed)

        # Apply button
        def apply_strategy():
            col = col_box.get()
            strat = strategy_box.get()
            val = custom_value.get()

            if not col or not strat:
                messagebox.showerror("Error", "Select a column and a strategy.")
                return

            try:
                if strat == "Drop Rows":
                    self.df_clean = self.df_clean.dropna(subset=[col])
                elif strat == "Fill with Min":
                    self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].min())
                elif strat == "Fill with Max":
                    self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].max())
                elif strat == "Fill with Mean":
                    self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].mean())
                elif strat == "Fill with Median":
                    self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].median())
                elif strat == "Fill with Mode":
                    self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].mode()[0])
                elif strat == "Fill with Custom Value":
                    typed_val = float(val)
                    self.df_clean[col] = self.df_clean[col].fillna(typed_val)
                
                self.populate_comboboxes()
                self.show_stats()
                messagebox.showinfo("Done", f"Missing values in '{col}' handled with '{strat}' strategy.")
                missing_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(missing_window, text="Apply", command=apply_strategy).pack(pady=10)
   
    def strategy_changed(self, event):
        """Enable/Disable Min-Max entry fields based on selected normalization strategy."""
        if self.norm_strategy.get() == "Min-Max Normalization":
            self.min_entry.configure(state='normal')
            self.max_entry.configure(state='normal')
        else:
            self.min_entry.configure(state='disabled')
            self.max_entry.configure(state='disabled')

    def apply_normalization(self):
        """Apply the selected normalization strategy to the selected column."""
        if self.df_clean is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        col = self.norm_columns.get()
        if not col:
            messagebox.showerror("Error", "Please select a column to normalize.")
            return

        strategy = self.norm_strategy.get()
        try:
            if strategy == "Min-Max Normalization":
                min_value = float(self.min_entry.get())
                max_value = float(self.max_entry.get())
                if min_value >= max_value:
                    messagebox.showerror("Error", "Min value must be less than Max value.")
                    return

                # Apply Min-Max normalization
                self.df_clean[col] = (self.df_clean[col] - self.df_clean[col].min()) / (self.df_clean[col].max() - self.df_clean[col].min()) * (max_value - min_value) + min_value

            elif strategy == "Z-Score Normalization":
                # Apply Z-Score normalization
                self.df_clean[col] = (self.df_clean[col] - self.df_clean[col].mean()) / self.df_clean[col].std()

            self.show_stats()  # Update stats after normalization
            self.show_data()
            messagebox.showinfo("Done", f"Normalization applied to column '{col}' using '{strategy}' strategy.")
        
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def run_clustering(self):
        """Run selected clustering algorithm and show results"""
        if self.df_clean is None:
            messagebox.showerror("Error", "No data loaded")
            return
            
        try:
            # Get numeric data
            numeric_data = self.df_clean.select_dtypes(include=['number'])
            if len(numeric_data.columns) < 2:
                messagebox.showerror("Error", "Need at least 2 numeric columns")
                return
                
            X = StandardScaler().fit_transform(numeric_data)
            feature_names = numeric_data.columns.tolist()
            algo = self.cluster_algo.get()
            
            if algo == "K-Means":
                k = int(self.n_clusters.get())
                model = KMeans(n_clusters=k, random_state=16, n_init=10)
            elif algo == "K-Medoids":
                k = int(self.n_clusters.get())
                model = KMedoids(n_clusters=k, random_state=16)
            elif algo == "DBSCAN":
                eps = float(self.dbscan_eps.get())
                min_samples = int(self.dbscan_min_samples.get())
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algo == "DIANA":
                k = int(self.n_clusters.get())
                model = DIANA(n_clusters=k)
            else:
                raise ValueError("Select an algorithm first")
                
            # Fit model and get clusters
            clusters = model.fit_predict(X)
            self.df_clean['Cluster'] = clusters  # Add to dataframe
            
            # Show cluster visualization window
            ClusterVisualizationWindow(self.root, X, clusters, feature_names, algo)
            
            # Show cluster info
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(clusters == -1) if -1 in unique_clusters else 0
            
            message = f"{algo} clustering complete!\n"
            message += f"Number of clusters: {n_clusters}\n"
            if algo == "DBSCAN":
                message += f"Number of noise points: {n_noise}\n"
            message += "Cluster labels added to dataset."
            
            messagebox.showinfo("Success", message)
                            
        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed:\n{str(e)}")
    

if __name__ == "__main__":
    root = tk.Tk()
    app = DataMiningApp(root)
    root.mainloop()

