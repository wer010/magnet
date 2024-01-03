import numpy as np
import torch


class magnetic_field:
    def __init__(self):
        pass

    def get_magnetic_field(self,p):
        return self.get_mfield()

    def get_mfield(self):
        raise NotImplementedError