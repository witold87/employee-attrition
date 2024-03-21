import pandas as pd
import numpy as np
from v2.data_transformers.BaseTrsTransformer import BaseTrsTransformer
import logging
from v2.mappers import grade, source_to_cat, names_exclusion_list, clients, projects, technology, job_family

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AttritionTrsTransformer(BaseTrsTransformer):

    def __init__(self, path_to_data, cleanup=False, predict: bool = False):
        super().__init__(path_to_data)
        self.predict = predict
        self.data = super().prepare_data(encrypt=False,
                                         save=False,
                                         save_mapping_table=False,
                                         cleanup=cleanup)
        logger.info(f'Loaded data...')

    def _use_related_employee_statuses(self):
        # self.data = self.data.loc[(self.data['Status'] == 'Employee') | (self.data['Status'] == 'Notice period')]
        # self.data = self.data.query('Status == Employee or Status == Notice period')
        self.data = self.data[self.data['status'] == 'Employee']
        logger.info(f'Using related statuses...')

    def _use_related_division(self):
        self.data = self.data.loc[(self.data['division'] == 'Delivery')]
        logger.info(f'Using related division...')

    def _cleanup_for_attrition(self):
        columns_to_remove = ['position',
                             'end date',
                             'traffic source',
                             'brand awareness source',
                             'owner',
                             'division',
                             'contract type',
                             'office location'
        ]

        additional_columns_to_remove = ['no', 'status', 'grade',
                                        'technology',
                                        'start date',
                                        #'client',
                                        'current project', 'report_date', 'project status', 'source',
                                        'job family',
                                        'report_date_dt',
                                        'start_date_dt',
                                        'max_date', 'max_report_date_all', 'diff',
                                        'initial_grade',
                                        'last_grade',
                                        'mapped_source',
                                        'last_client',
                                        'last_project',
                                        'new_job_family',
                                        'technology'
                                        ]
        logger.info(f'Removing columns...')
        self.data.drop(columns_to_remove, axis=1, inplace=True, errors='ignore')
        self.data.drop(additional_columns_to_remove, axis=1, inplace=True, errors='ignore')

    def _get_max_date(self, x: pd.Series):
        return np.array([x.max() for _ in range(x.shape[0])])

    def _transform_to_datetime(self, column_name: str):
        logger.info('Creating datetime column...')
        new_dt_column = f'{column_name}_dt'
        if ' ' in column_name:
            new_dt_column = new_dt_column.replace(' ', '_').lower()
        self.data[new_dt_column] = pd.to_datetime(self.data[column_name], format='%d.%m.%Y')

    def _set_max_date(self):
        assert 'report_date_dt' in self.data.columns
        self.data['max_date'] = self.data.groupby('unique_id')['report_date_dt'].transform(self._get_max_date)

    def _transform_to_dummies(self):
        columns_to_dummies = [
            'office location',
            'contract type',
            'new_job_family',
            'mapped_source',
            #'tenure_group',
            'technology'
        ]

        logger.info(f'Creating dummies for columns {columns_to_dummies}')
        dummies_df = pd.concat([pd.get_dummies(self.data[column_name]) for column_name in columns_to_dummies], axis=1)
        assert dummies_df.shape[0] == self.data.shape[0]
        self.data = pd.concat([self.data, dummies_df], axis=1)

    def _get_last_grade(self, x: pd.Series):
        grades = x.unique().tolist()
        n_grades = len(grades)
        result = sorted(grades)[n_grades - 1]
        return np.array([result for _ in range(x.shape[0])])

    def _set_last_grade(self):
        self.data['last_grade'] = self.data.groupby('unique_id')['grade'].transform(self._get_last_grade)

    def _get_initial_grade(self, x: pd.Series):
        result = sorted(x.unique().tolist())[0]
        return np.array([result for _ in range(x.shape[0])])

    def _set_initial_grade(self):
        self.data['initial_grade'] = self.data.groupby('unique_id')['grade'].transform(self._get_initial_grade)

    def _calculate_tenure(self):
        assert 'max_date' in self.data.columns and 'start_date_dt' in self.data.columns
        self.data['tenure_in_years'] = (self.data['max_date'] - self.data['start_date_dt']).dt.days / 365

    def _get_n_promotions(self, x: pd.Series):
        result = len(set(x.unique().tolist())) - 1  # two unique values means n - 1 promotions
        return np.array([result for _ in range(x.shape[0])])

    def _calculate_promotions(self):
        self.data['n_promotions'] = self.data.groupby('unique_id')['grade'].transform(self._get_n_promotions)

    def _get_count(self, x: pd.Series):
        result = len(set([str(item).lower() for item in x.unique().tolist()]))
        return np.array([result for _ in range(x.shape[0])])

    def _get_n_count_for_column(self, column):
        if column == 'client':
            new_column = 'n_clients'
        else:
            new_column = 'n_projects'
        self.data[new_column] = self.data.groupby('unique_id')[column].transform(self._get_count)

    def _calculate_if_stayed(self):
        self.data['diff'] = (self.data['max_report_date_all'] - self.data['max_date']).dt.days

    def _has_left(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def _calculate_target_variable(self):
        self._calculate_if_stayed()
        has_left_df = self.data['diff'].apply(self._has_left)
        self.data['stay/left'] = has_left_df

    def map_grades(self, x):
        if str(x) in grade.keys():
            return grade[str(x)]

    def _remap_grades(self, grade_level):
        grades_mapped = self.data[grade_level].apply(self.map_grades)
        self.data[f'mapped_{grade_level}'] = grades_mapped

    def _map_job_family(self, x):
        if str(x) in job_family.keys():
            return job_family[str(x)]
        else:
            return job_family['other']

    def _remap_job_family(self):
        job_family_mapped = self.data['job family'].apply(self._map_job_family)
        self.data['mapped_job_family'] = job_family_mapped

    def _map_source(self, x):
        for source_key, source_list in source_to_cat.items():
            if str(x) in source_list:
                return str(source_key)

    def _remap_source(self):
        source_mapped = self.data['source'].apply(self._map_source)
        self.data['mapped_source'] = source_mapped

    def _map_client(self, x):
        if str(x) in clients.keys():
            return clients[str(x)]
        else:
            return clients['other']

    def _remap_last_client(self):
        client_mapped = self.data['last_client'].apply(self._map_client)
        self.data['mapped_last_client'] = client_mapped

    def _map_last_project(self, x):
        if str(x) in projects.keys():
            return projects[str(x)]
        else:
            return projects['other']

    def _remap_last_project(self):
        client_mapped = self.data['last_project'].apply(self._map_last_project)
        self.data['mapped_last_project'] = client_mapped

    def _calc_avg_time_per_proj(self):
        assert 'tenure_in_years' in self.data.columns and 'n_projects' in self.data.columns
        self.data['avg_time_per_proj'] = self.data['tenure_in_years'] / self.data['n_projects']

    def _use_related_project_status(self):
        self.data = self.data[self.data['project status'] == 'In project']

    def _get_max_report_date_for_all(self):
        assert 'report_date_dt' in self.data.columns
        self.data['max_report_date_all'] = self.data['report_date_dt'].max()

    def _drop_duplicates(self):
        self.data.drop_duplicates(subset='unique_id', keep='last', inplace=True)

    def _check_if_promoted(self, x):
        n_promotions = x.unique().tolist()[0]
        promoted = 0
        if n_promotions > 0:
            promoted = 1
        return np.array([promoted for _ in range(x.shape[0])])

    def _was_promoted(self):
        self.data['was_promoted'] = self.data.groupby('unique_id')['n_promotions'].transform(self._check_if_promoted)

    def _sort_data(self):
        assert 'report_date_dt' in self.data.columns
        self.data.sort_values(['report_date_dt'], ascending=True, inplace=True)

    def _set_gender(self):
        self.data['gender'] = self.data['name'].apply(
            lambda x: 1 if x.endswith('a') and x not in names_exclusion_list else 0)

    def _modify_families(self, x):
        if str(x) == 'UX':
            return 'Digital'
        elif str(x) == 'DevOps':
            return 'AMS'
        else:
            return str(x)

    def _map_reduce_families(self, x):
        if str(x) in job_family.keys():
            return str(x)
        else:
            return 'other'

    def _change_job_family_input(self):
        logger.info(f'Changing the family job input due to data inconsistency...')
        modified_job_family = self.data['job family'].apply(self._modify_families)
        self.data['new_job_family'] = modified_job_family
        mapped_families = self.data['new_job_family'].apply(self._map_reduce_families)
        self.data['new_job_family'] = mapped_families

    def _use_related_dates(self, min_date):
        logger.info(f'Using related dates...')
        mind_date_dt = pd.to_datetime(min_date)
        self.data = self.data.loc[(self.data['report_date_dt'] >= mind_date_dt)]

    def _calc_avg_promotion_per_tenure(self):
        assert 'n_promotions' in self.data.columns and 'tenure_in_years' in self.data.columns
        self.data = self.data.assign(avg_prom_time=lambda x: x['tenure_in_years'] / x['n_promotions'])

    def _calc_bench_time_tenure(self):
        assert 'time_on_bench_year' in self.data.columns and 'tenure_in_years' in self.data.columns
        self.data = self.data.assign(bench_tenure_time=lambda x: x['time_on_bench_year'] / x['tenure_in_years'])

    def _fill_missing_data_for_projects(self):
        self.data.loc[self.data['client'] == 'Bench', 'project status'] = 'In project'

    def _fill_missing_data_for_bench(self):
        self.data.loc[
            (self.data['client'] == 'Bench') & (self.data['current project'].isnull()), 'current project'] = 'Bench'
        self.data.loc[self.data['current project'].isnull(), 'current project'] = 'nd'

    def _get_last_value(self, x: pd.Series):
        last_row = x.iloc[-1:]
        project = last_row.values[0]
        return np.array([project for _ in range(x.shape[0])])

    def _get_last_project(self):
        self.data['last_project'] = self.data.groupby('unique_id')['current project'].transform(self._get_last_value)

    def _get_last_client(self):
        self.data['last_client'] = self.data.groupby('unique_id')['client'].transform(self._get_last_value)

    def _get_bench_time(self, x):
        from operator import countOf
        project_list = list(x.values)
        n_bench_occurences = countOf(project_list, 'Bench')  # each record equals one month
        time_as_year = float(n_bench_occurences / 12)
        return np.array([time_as_year for _ in range(x.shape[0])])

    def _calc_bench_time(self):
        self.data['time_on_bench_year'] = self.data.groupby('unique_id')['client'].transform(self._get_bench_time)

    def _map_technology(self, x):
        if str(x) in technology.keys():
            return f'tech_{str(x)}'
        else:
            return 'tech_other'

    def _get_technology(self):
        technology_mapped = self.data['technology'].apply(self._map_technology)
        self.data['technology'] = technology_mapped

    def _map_tenure(self, x):
        if x <= 1:
            return 't-1'
        elif 1 < x <= 2:
            return '1-t-2'
        elif 2 < x <= 3:
            return '2-t-3'
        elif 3 < x <= 4:
            return '3-t-4'
        elif 4 < x <= 5:
            return '4-t-5'
        else:
            return 't-5'

    def _get_tenure_group(self):
        technology_mapped = self.data['tenure_in_years'].apply(self._map_tenure)
        self.data['tenure_group'] = technology_mapped

    def _normalize_tenure(self): # SHITTY RESULTS
        # from sklearn import preprocessing
        # tenure = self.data['tenure_in_years'].values.reshape((-1, 1))
        # self.data['tenure_normalized'] = preprocessing.normalize(tenure)

        # from sklearn import preprocessing
        # x = self.data['tenure_in_years'].values  # returns a numpy array
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x_scaled = min_max_scaler.fit_transform(x)
        # self.data['tenure_normalized'] = pd.DataFrame(x_scaled)
        self.data['tenure_normalized'] = self.data.apply(lambda x: (x['tenure_in_years'] - x['tenure_in_years'].mean()) / x['tenure_in_years'].std())


    def prepare_data_for_attrition_prediction(self, min_date):
        self._use_related_employee_statuses()
        self._fill_missing_data_for_projects()
        self._fill_missing_data_for_bench()
        self._transform_to_datetime('report_date')
        self._sort_data()
        self._transform_to_datetime('start date')
        self._set_gender()
        self._use_related_division()
        self._get_technology()
        self._set_max_date()
        self._get_max_report_date_for_all()
        self._set_initial_grade()
        self._set_last_grade()
        self._calculate_promotions()
        self._was_promoted() # -> feature irrelevant, we have n_promotions
        self._calculate_tenure()
        #self.tenure_in_account()
        #self.tenure_in_project()
        #self.tenure_on_bench_last_6m() # before leaving if 1
        self._get_n_count_for_column('client')
        self._get_n_count_for_column('current project')
        self._remap_grades('initial_grade')
        self._remap_grades('last_grade')
        self._remap_source()
        self._change_job_family_input()
        #self._get_tenure_group()
        #self._normalize_tenure()
        self._transform_to_dummies()
        self._calc_avg_time_per_proj()
        #if not self.predict:
        self._calculate_target_variable()
        self._use_related_dates(min_date)
        self._get_last_project()
        self._get_last_client()
        self._remap_last_client()
        self._remap_last_project()
        self._calc_bench_time()
        self._cleanup_for_attrition()
        #self.data.drop(['n_promotions'], axis=1, inplace=True)  # testing with n_promotions
        #self.data.drop(['tenure_in_years'], axis=1, inplace=True)  # testing with tenure_in_years
        #self.data.drop(['tenure_group'], axis=1, inplace=True)  # testing with tenure_in_years
        self._drop_duplicates()
        return self.data
