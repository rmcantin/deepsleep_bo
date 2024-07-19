from enum import Enum


class DeviceClass(Enum):
    bbt = '_dev_bbt'
    et_tobii_fixed = '_dev_et_tobii_fixed'
    et_tobii_glasses_2 = '_dev_et_tobii_glasses_2'
    et_tobii_glasses_3 = '_dev_et_tobii_glasses_3'


hw_type_map_dict = {
    ('wifi_ring', 'bth_ring', 'ring', 'wifi_diadem_ec', 'bth_diadem_ec',
     'diadem', 'elevvo_diadem', 'bth_cap_08', 'bth_cap_16', 'bth_cap_32',
     'cap16', 'cap32', 'bth_diadem_b', 'bth_motor_cap', 'sub_motor_cap',
     'ble_cap_04', 'ble_cap_05', 'bth_cap_64', 'bth_diadem_m',
     'bth_baby_cap_16', 'bth_baby_cap_32',
     'bth_biosensing', 'Biosensing', 'bth_imu', 'ble_uwb',
     'ble_uwb_05', 'artificial',): DeviceClass.bbt,
    ('et_tobii_fixed', 'ET_Fixed'): DeviceClass.et_tobii_fixed,
    ('et_tobii_glasses', 'ET_Glasses'): DeviceClass.et_tobii_glasses_2,
    ('et_tobii_glasses_3', ): DeviceClass.et_tobii_glasses_3
}

hw_type_dict = {key: v for k, v in hw_type_map_dict.items() for key in k}


def is_bbt(device_type: str) -> bool:
    return is_class(device_type, DeviceClass.bbt)


def is_class(device_type: str, device_class: DeviceClass) -> bool:
    return hw_type_dict[device_type] == device_class
