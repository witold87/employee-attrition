import lime
from lime import lime_tabular
import numpy as np
import re
import pandas as pd
from tqdm import tqdm


def setup_explainer_on_train_data(training_data: pd.DataFrame):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(training_data),
        feature_names=training_data.columns,
        class_names=['low risk', 'high risk'],
        mode='classification'
    )
    return explainer


def prepare_contributors_for_prediction(pred_df: pd.DataFrame, classifier, explainer):
    all_contributors = []
    n_predictions = pred_df.shape[0]
    print(f'Processing {n_predictions} predictions')
    for i in tqdm(range(n_predictions)):
        exp = explainer.explain_instance(
            data_row=pred_df.iloc[i],
            predict_fn=classifier.predict_proba
        )
        individual_contributors = exp.as_list()
        individual_contributors = dict((re.sub('[^a-zA-Z_]', '', key), val) for (key, val) in individual_contributors)
        temp_dict = {i: individual_contributors}
        contrib_single_df = pd.DataFrame(temp_dict).transpose().reset_index()
        all_contributors.append(contrib_single_df)
    print(f'Processed successfully {len(all_contributors)} predictions')
    return all_contributors


