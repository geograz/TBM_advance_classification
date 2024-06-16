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
            case 'Ulriken':
                self.param_dict = {'filename': 'synth_df_UT_4096',
                                   'plotname': 'sample_Ulriken',
                                   'n cutters': 50,
                                   'cutter radius': 241.3,  # [mm]
                                   'cutterhead diameter': 8,  # [m]
                                   'cutterhead rotations': 3.5,  # [rpm]
                                   'M0': 400,  # [kNm]
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 5  # frequency of data recording
                                   }
            case 'BBT':
                self.param_dict = {'filename': 'synth_df_BBT_4096',
                                   'plotname': 'sample_BBT',
                                   'n cutters': 50,
                                   'cutter radius': 216,  # [mm]
                                   'cutterhead diameter': 6,  # [m]
                                   'cutterhead rotations': 3.5,  # [rpm]
                                   'M0': 300,  # [kNm]
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 5  # frequency of datapoint recording [s]
                                   }
            case 'Follo':
                self.param_dict = {'filename': 'synth_df_FB_4096',
                                   'plotname': 'sample_Follo',
                                   'n cutters': 50,
                                   'cutter radius': 216,  # [mm]
                                   'cutterhead diameter': 6,  # [m]
                                   'cutterhead rotations': 3.5,  # [rpm]
                                   'M0': 400,  # [kNm]
                                   'stroke length': 1.7,  # [m]
                                   'frequency': 5  # frequency of datapoint recording [s]
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
