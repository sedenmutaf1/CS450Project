import os
import shutil

class StorageSimulator:
    def __init__(self, base_dir="storage"):
        self.base_dir = os.path.abspath(base_dir)
        self.reset_metrics()
        
        # Ensure directories exist
        for folder in ["videos", "frames", "clips", "outputs"]:
            os.makedirs(os.path.join(self.base_dir, folder), exist_ok=True)

    def reset_metrics(self):
        self.put_requests = 0
        self.get_requests = 0
        self.bytes_uploaded = 0
        self.bytes_downloaded = 0

    def _resolve_path(self, key):
        # Prevent directory traversal
        clean_key = key.replace("\\", "/").lstrip("/")
        return os.path.join(self.base_dir, clean_key)

    def put_file(self, src_path, key):
        """Simulates uploading a file to S3."""
        dest_path = self._resolve_path(key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file if it exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            size = os.path.getsize(dest_path)
        else:
            # If mocking, create a dummy file
            with open(dest_path, "wb") as f:
                f.write(b"mock_data")
            size = 9
            
        self.put_requests += 1
        self.bytes_uploaded += size
        return key

    def get_file(self, key, dest_path):
        """Simulates downloading a file from S3."""
        src_path = self._resolve_path(key)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"File not found in storage: {key}")
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        
        size = os.path.getsize(src_path)
        self.get_requests += 1
        self.bytes_downloaded += size
        return dest_path

    def write_data(self, data, key):
        """Simulates writing raw bytes or string directly to S3."""
        dest_path = self._resolve_path(key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        with open(dest_path, "wb") as f:
            f.write(data)
            
        self.put_requests += 1
        self.bytes_uploaded += len(data)
        return key

    def read_data(self, key):
        """Simulates reading raw bytes directly from S3."""
        src_path = self._resolve_path(key)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"File not found in storage: {key}")
            
        with open(src_path, "rb") as f:
            data = f.read()
            
        self.get_requests += 1
        self.bytes_downloaded += len(data)
        return data

    def list_files(self, prefix=""):
        """Lists file keys in simulated S3 starting with prefix."""
        search_dir = self._resolve_path(prefix)
        if not os.path.exists(search_dir):
            # Check if parent exists and filter by prefix
            parent_dir = os.path.dirname(search_dir)
            if not os.path.exists(parent_dir):
                return []
            search_dir = parent_dir
            
        keys = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.base_dir)
                key = rel_path.replace("\\", "/")
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete_folder(self, folder_prefix):
        """Deletes all files under a simulated folder/prefix."""
        folder_path = self._resolve_path(folder_prefix)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

    def calculate_cost(self):
        """
        Calculates simulated AWS S3 cost based on:
        - PUT requests: $0.005 per 1,000 requests ($0.000005 per PUT)
        - GET requests: $0.0004 per 1,000 requests ($0.0000004 per GET)
        - Data Transfer Out: $0.09 per GB ($0.00000009 per byte)
        """
        put_cost = self.put_requests * 0.000005
        get_cost = self.get_requests * 0.0000004
        transfer_cost = self.bytes_downloaded * 0.00000009
        return put_cost + get_cost + transfer_cost

    def get_metrics(self):
        return {
            "put_requests": self.put_requests,
            "get_requests": self.get_requests,
            "bytes_uploaded": self.bytes_uploaded,
            "bytes_downloaded": self.bytes_downloaded,
            "total_cost_usd": self.calculate_cost()
        }
