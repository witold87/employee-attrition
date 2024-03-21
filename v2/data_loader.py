import pandas as pd
import json
import io
import msoffcrypto


with open('data_catalog.json', 'r') as file:
    data_catalog = json.load(file)
    assert type(data_catalog) == dict


def _decrypt_file(data_source_type, subtype: str):
    base_file_path = data_catalog.get('general_path')
    if subtype == 'perm':
        path = data_catalog.get(data_source_type).get('perm').get('file')
        passwd = data_catalog.get(data_source_type).get('perm').get('pass')
    elif subtype == 'b2b':
        path = data_catalog.get(data_source_type).get('b2b').get('file')
        passwd = data_catalog.get(data_source_type).get('b2b').get('pass')
    else:
        raise Exception(f'Cannot find a data for subtype {subtype}')
    decrypted_workbook = io.BytesIO()

    final_path = f'{base_file_path}/{path}'

    with open(final_path, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=passwd)
        office_file.decrypt(decrypted_workbook)
    return decrypted_workbook


def _load_encrypted_data(data_source_type, subtype):
    decrypted_file = _decrypt_file(data_source_type, subtype)
    data = pd.read_excel(decrypted_file)
    return data


def load_data(data_source_type: str,
              subtype: str = None):
    base_path = data_catalog.get('general_path')

    if data_source_type == 'finance' and subtype is not None:
        data = _load_encrypted_data(data_source_type, subtype)
    else:
        file_path = data_catalog.get(data_source_type).get('file')
        final_path = f'{base_path}/{file_path}'
        data = pd.read_excel(final_path)
    return data

