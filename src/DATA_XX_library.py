# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:29:38 2024

@author: GEr
"""

import numpy as np
import pandas as pd


class utilities:

    def __init__(self, SAMPLE):
        # dataset specific metadata dictionaries
        match SAMPLE:  # noqa
            case 'TBM_A':
                self.param_dict = {'filename': 'TBM_A',
                                   'n cutters': 50,
                                   'cutter radius': 241.3,  # [mm]
                                   'cutterhead diameter': 7,  # [m]
                                   'cutterhead rotations': 6,  # [rpm]
                                   'M0': 180,  # [kNm]
                                   'torque ratio bounds': (0.35, 0.57),
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 10  # frequency of datapoint recording [s]
                                   }
            case 'TBM_B':
                self.param_dict = {'filename': 'TBM_B',
                                   'n cutters': 70,
                                   'cutter radius': 241.3,  # [mm]
                                   'cutterhead diameter': 10,  # [m]
                                   'cutterhead rotations': 5,  # [rpm]
                                   'M0': 400,  # [kNm]
                                   'torque ratio bounds': (0.45, 0.57),
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 1  # frequency of datapoint recording [s]
                                   }
            case 'TBM_C':
                self.param_dict = {'filename': 'TBM_C',
                                   'n cutters': 60,
                                   'cutter radius': 241.3,  # [mm]
                                   'cutterhead diameter': 9,  # [m]
                                   'cutterhead rotations': 5,  # [rpm]
                                   'M0': 220,  # [kNm]
                                   'torque ratio bounds': (0.53, 0.63),
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 10  # frequency of data recording
                                   }

    def torque_ratio(self,
                     advance_force: pd.Series,  # cutterhead advance force [kN]
                     penetration: pd.Series,  # penetration [mm/rev]
                     torque: pd.Series  # real torque [kNm]
                     ) -> pd.Series:
        '''computation of the torque ratio according to the Austrian
        contractual standard: ÖNORM B2203-2 (2023)'''
        F_n = advance_force / self.param_dict['n cutters']
        alpha = np.arccos((self.param_dict['cutter radius'] - penetration)/self.param_dict['cutter radius'])
        F_tang = np.tan(alpha/2)*F_n
        theo_torque = 0.3 * F_tang * self.param_dict['n cutters'] * self.param_dict['cutterhead diameter'] + self.param_dict['M0']
        torque_ratio = torque / theo_torque
        return theo_torque, torque_ratio
