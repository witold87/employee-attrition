import pandas as pd
import numpy as np
import uuid
import logging
from v2.exceptions.data_exceptions import ColumnNotFoundException
import unidecode
from v2.utils.crypto import encrypt_text
import datetime

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


mandatory_columns = [
    'Name',
    'Surname',
    'Office location',
    'Start date'
]

useless_columns = ['unnamed: 0', 'telephone', 'email', 'skype', 'prefix', 'provisional booking status',
                   'provisional booking client',
                   'provisional booking project',
                   'tentative assigned status',
                   'tentative assigned client',
                   'tentative assigned project']


class BaseTrsTransformer():

    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data
        self.data = self._load_and_validate()

    def _load_and_validate(self):
        logger.info(f'Loading data from {self.path_to_data}')
        if self.path_to_data[-4] == '.csv':
            self.data = pd.read_csv(self.path_to_data, sep=';', encoding='UTF8')
        else:
            self.data = pd.read_excel(self.path_to_data, sheet_name='Sheet1')
        self._validate_data()
        self._consolidate_data()
        return self.data

    def _consolidate_data(self):
        logging.info(f'Changing column names in data...')
        self.data.columns = self.data.columns.str.lower()
        self.data['status'] = self.data['status'].str.lower()


    def _validate_data(self):
        logging.info(f'Validating data...')
        for col in mandatory_columns:
            if col not in self.data.columns:
                raise ColumnNotFoundException(f'Mandatory column - {col} - not found in data file')

    def _pre_clean_data(self):
        self.data.drop(useless_columns, axis=1, inplace=True, errors='ignore')

    def _add_employee_name_column(self):
        logger.info('Employee column being generated...')
        self.data['employee'] = self.data['name'] + ' ' + self.data['surname']

    def _decode_polish_chars(self):
        logger.info('Decoding polish characters...')
        self.data['employee'] = self.data['employee'].astype(str)
        self.data['office location'] = self.data['office location'].astype(str)
        self.data['employee'] = self.data['employee'].apply(lambda x: unidecode.unidecode(x))
        self.data['office location'] = self.data['office location'].apply(lambda x: unidecode.unidecode(x))

    def _add_unique_identifier(self):
        logger.info('Adding unique identifier...')
        self.data['start date'] = self.data['start date'].astype(str)
        self.data['start date'] = self.data['start date'].apply(lambda x: x[0:10])
        grouped = self.data.groupby(['employee', 'office location', 'start date'])
        ngroups = grouped.ngroups
        logger.debug(f'Identified {ngroups} groups (unique identifiers)')
        uuids = np.array([str(uuid.uuid4()) for _ in range(ngroups)])
        self.data['unique_id'] = uuids[grouped.ngroup()]

    def _encrypt_data(self, data: pd.Series):
        entity_to_encrypt = str(data.values.tolist()[0])
        encrypted = encrypt_text(entity_to_encrypt)
        return np.array([encrypted for _ in range(data.shape[0])])

    def _encrypt_sensitive_data(self):
        logger.info('Encrypting data...')
        self.data['employee_encrypted'] = self.data.groupby(['employee', 'office location', 'start date'])['employee'].transform(
            self._encrypt_data)

    def _clean_up(self):
        self.data.drop(['name', 'surname', 'employee'], axis=1, inplace=True, errors='ignore')

    def _clean_technology(self):
        logger.info(f'Cleaning technology...')
        searched_word = 'Warsaw'
        self.data['technology'] = self.data['technology'].apply(lambda x: x.replace(searched_word, '').strip() if searched_word in x else x)

    def _save_data(self, output_path: str = '<path>', save_mapping_table: bool = True):
        logger.info(f'Saving data to {output_path}...')
        today = datetime.datetime.now().strftime('%d-%m-%Y')
        if save_mapping_table:
            logging.info(f'Saving mapping data...')
            mapping_df = self.data[['unique_id', 'employee_encrypted']]
            mapping_df.to_csv(f'{output_path}/mapping_table_{today}.csv', sep=';', encoding='UTF8')
            assert mapping_df.shape[0] == self.data.shape[0]
        self.data.drop(['employee_encrypted'], axis=1, inplace=True, errors='ignore')
        self.data.to_csv(f'{output_path}/data_{today}.csv', sep=';', encoding='UTF8')
        logger.info('Saving complete.')

    def prepare_data(self, encrypt: bool = False,
                     save: bool = True,
                     output_path: str = None,
                     cleanup: bool = True,
                     save_mapping_table: bool = True):
        self._pre_clean_data()
        self._add_employee_name_column()
        self._decode_polish_chars()
        self._add_unique_identifier()
        self._clean_technology()

        if encrypt:
            self._encrypt_sensitive_data()

        if cleanup:
            self._clean_up()

        if save:
            self._save_data(output_path, save_mapping_table)

        logger.info(f'Data processed: shape {self.data.shape}, columns {self.data.columns}')
        return self.data