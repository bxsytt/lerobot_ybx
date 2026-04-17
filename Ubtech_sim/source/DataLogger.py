import csv
import h5py
import numpy as np
from pathlib import Path


class DataLogger:
    """Handles time-series logging of pose data to CSV files and camera RGB data to HDF5 files."""
    
    def __init__(self, enabled=False, csv_path="poses.csv", camera_enabled=False, camera_hdf5_path="camera_data.hdf5"):
        """
        Initialize the pose data logger.
        
        Args:
            enabled: Whether CSV pose logging is enabled
            csv_path: Path to save the CSV file
            camera_enabled: Whether HDF5 camera logging is enabled
            camera_hdf5_path: Path to save the HDF5 camera file
        """
        self.enabled = enabled
        self.csv_path = csv_path
        self.header_written = False
        self.time_step = 0
        
        # Camera logging configuration
        self.camera_enabled = camera_enabled
        self.camera_hdf5_path = camera_hdf5_path
        self.camera_hdf5_file = None
        self.camera_step_count = 0
        
        if self.enabled:
            # Ensure directory exists for CSV
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.camera_enabled:
            # Ensure directory exists for HDF5
            Path(self.camera_hdf5_path).parent.mkdir(parents=True, exist_ok=True)
            self.camera_hdf5_file = h5py.File(self.camera_hdf5_path, 'w')
    
    def log_poses(self, poses_data):
        """Append current pose data to CSV file with time step.
        
        Args:
            poses_data: Dictionary with prim_path as key and pose info as value
        """
        if not self.enabled or not poses_data:
            return
            
        # Prepare fieldnames and row data
        fieldnames = ['time_step']
        row_data = [self.time_step]
        
        # Add fields for each prim path in the poses_data
        for prim_path, pose_info in poses_data.items():
            module_name = pose_info['module_name']
            # Create headers for each pose component
            fieldnames.extend([
                f"{module_name}_{prim_path}_x",
                f"{module_name}_{prim_path}_y", 
                f"{module_name}_{prim_path}_z",
                f"{module_name}_{prim_path}_qx",
                f"{module_name}_{prim_path}_qy", 
                f"{module_name}_{prim_path}_qz", 
                f"{module_name}_{prim_path}_qw"
            ])
            row_data.extend([
                pose_info['world_position'][0],
                pose_info['world_position'][1],
                pose_info['world_position'][2],
                pose_info['world_orientation'][0],
                pose_info['world_orientation'][1],
                pose_info['world_orientation'][2],
                pose_info['world_orientation'][3]
            ])
        
        # Write header if not already written
        if not self.header_written:
            file_exists = Path(self.csv_path).exists() and Path(self.csv_path).stat().st_size > 0
            if not file_exists:
                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(fieldnames)
            self.header_written = True
        
        # Append data row
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        
        # Increment time step
        self.time_step += 1
    
    def log_camera_rgb(self, camera_data):
        """Log RGB data from cameras to HDF5 file.
        
        Args:
            camera_data: Dictionary with camera_name as key and RGB data as value
        """
        if not self.camera_enabled or not camera_data or not self.camera_hdf5_file:
            return
            
        try:
            step_group = self.camera_hdf5_file.create_group(f'step_{self.camera_step_count}')
            
            for camera_name, data in camera_data.items():
                if data and 'rgb' in data and data['rgb'] is not None:
                    # Convert RGB data to numpy array if it isn't already
                    rgb_array = np.array(data['rgb'])
                    step_group.create_dataset(f'{camera_name}_rgb', data=rgb_array, compression='gzip')
            
            # Flush periodically to ensure data is written to disk
            if self.camera_step_count % 10 == 0:
                self.camera_hdf5_file.flush()
                
            self.camera_step_count += 1
        except Exception as e:
            print(f"Error logging camera data: {e}")
    
    def close(self):
        """Close all logging files properly."""
        if self.camera_enabled and self.camera_hdf5_file and self.camera_hdf5_file.id.valid:
            try:
                self.camera_hdf5_file.flush()
                self.camera_hdf5_file.close()
                print(f"HDF5 camera file saved successfully: {self.camera_hdf5_path}")
            except Exception as e:
                print(f"Error closing HDF5 camera file: {e}")