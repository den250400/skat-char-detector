import pandas as pd
import numpy as np
import random
from collections import deque

class CoordinateWriter:
    def __init__(self, path: str, lat: float, lon: float, maxlen: int = 10):
        self.path = path
        self.coordinate_dict = {
            'id': [0],
            'lat': [0],
            'lon': [0]
        }
        self.lat = lat
        self.lon = lon
        self.last_id_list = deque([-1 for _ in range(maxlen)], maxlen=5)

    def _write(self):
        df = pd.DataFrame.from_dict(self.coordinate_dict).iloc[1:, :]
        df.to_csv(self.path, index=False)

    def append(self, id: int):
        self.last_id_list.append(id)

        if (np.array(self.last_id_list) == self.last_id_list[-1]).all() and self.coordinate_dict['id'][-1] != id:
            self.coordinate_dict['id'].append(id)
            self.coordinate_dict['lat'].append(self.lat + random.randint(-1000, 1000))
            self.coordinate_dict['lon'].append(self.lon + random.randint(-1000, 1000))
            self._write()