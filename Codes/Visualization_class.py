import plotly.graph_objects as go
import numpy as np

class SensorGridVisualization:
    def __init__(self, sensor_width=10, sensor_height=8, grid_x_res=20, grid_y_res=25):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.grid_x_res = grid_x_res
        self.grid_y_res = grid_y_res
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, self.sensor_width, self.grid_x_res),
            np.linspace(0, self.sensor_height, self.grid_y_res)
        )
        self.sensor_z = np.full_like(self.grid_x, 0.7)

    def normalize_heatmap(self, heatmap_values):
        return (heatmap_values - np.min(heatmap_values)) / (np.max(heatmap_values) - np.min(heatmap_values))

    def visualize(self, heatmap_values, attractor_point, sensor_points=[[5, 2, 0.8], [5, 6, 0.8]], title=None, show_colorbar=True):
        normalized_values = self.normalize_heatmap(heatmap_values.reshape(self.grid_y_res, self.grid_x_res))

        smooth_colorscale = [
            [0.0, "rgb(0, 0, 255)"],   # Deep Blue
            [0.1, "rgb(0, 128, 255)"], # Light Blue
            [0.4, "rgb(0, 255, 128)"], # Cyan-Green
            [0.7, "rgb(255, 255, 0)"], # Yellow
            [0.9, "rgb(255, 128, 0)"], # Orange
            [1, "rgb(255, 0, 0)"]      # Red
        ]

        sensor_surface = go.Surface(
            z=self.sensor_z, x=self.grid_x, y=self.grid_y,
            surfacecolor=normalized_values,
            colorscale=smooth_colorscale,
            opacity=0.9,
            showscale=show_colorbar,
            colorbar=dict(title="Heatmap Values") if show_colorbar else None
        )

        attractor = go.Scatter3d(
            x=[attractor_point[0]], y=[attractor_point[1]], z=[attractor_point[2]],
            mode='markers',
            marker=dict(size=6, color='red', opacity=1),
            name='Attractor Point'
        )

        sensor_pts = go.Scatter3d(
            x=[p[0] for p in sensor_points],
            y=[p[1] for p in sensor_points],
            z=[p[2] for p in sensor_points],
            mode='markers',
            marker=dict(size=4, color='white', opacity=1),
            name='Sensor Points'
        )

        bg_y, bg_z = np.meshgrid(np.linspace(0, self.sensor_width, 10), np.linspace(0, 3, 10))
        bg_x = np.full_like(bg_y, 0)

        background_plane = go.Surface(
            x=bg_x, y=bg_y, z=bg_z,
            colorscale=[(0, "#99FDFF"), (1, "#ffffff")],
            opacity=0.5,
            showscale=False
        )

        fig = go.Figure(data=[sensor_surface, attractor, sensor_pts, background_plane])

        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=-10, y=-10, z=7),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(range=[0, self.sensor_width], showbackground=False, showticklabels=False, title=""),
                yaxis=dict(range=[0, self.sensor_height], showbackground=False, showgrid=True,
                           showticklabels=False, title=""),
                zaxis=dict(range=[0, 3], showbackground=True, showgrid=False,
                           gridcolor='#cecece', showticklabels=False, title=""),
                aspectmode='manual',
                aspectratio=dict(x=8, y=10, z=3)
            ),
            title=title or 'Scaled 3D Sensor Grid Visualization'
        )

        return fig
