import json
import pandas as pd
import os


class Parser:

    def __init__(self, path_to_datas):
        self.path_to_datas = path_to_datas
        self.files = os.listdir(path_to_datas)
        print(f'The following files will be parsed: {self.files}')

    # -----------------------------------------------------
    def raw_to_df(self):
        list_of_df = []
        for file in self.files:
            with open(f'{self.path_to_datas}/{file}', 'r') as f:
                data = json.loads(f.read())
            predicts = self._factor_mean(data, 'result')
            df_dict = self._create_df(data, predicts)
            df = pd.DataFrame(df_dict)
            list_of_df.append(df)
        df = pd.concat(list_of_df)
        return df
    # -----------------------------------------------------

    # -----------------------------------------------------
    def _choose_best(self, item):
        return max(set(item), key=item.count)
    # -----------------------------------------------------

    # -----------------------------------------------------
    def _create_df(self, inp, preds):
        dict_df = {'source_prompt': [],
                   'target_prompt': [],
                   'image_1': [],
                   'image_2': [],
                   'result': [],
                   'task_id': []}
        for el in inp:
            task_id = el["taskId"]
            if task_id in dict_df['task_id']:
                continue
            dict_df['task_id'].append(task_id)
            dict_df['image_1'].append(el['inputValues']["image_1"])
            dict_df['image_2'].append(el['inputValues']["image_2"])
            dict_df['source_prompt'].append(el['inputValues']["orig_text"])
            dict_df['target_prompt'].append(el['inputValues']["new_text"])
            dict_df['result'].append(preds[task_id])

        return dict_df
    # -----------------------------------------------------

    # -----------------------------------------------------
    def _factor_mean(self, inp, factor):
        vals = {}
        for el in inp:
            if el['outputValues'][factor] == 'error':
                continue
            try:
                vals[el["taskId"]].append(el['outputValues'][factor])
            except KeyError:
                vals[el["taskId"]] = []
                vals[el["taskId"]].append(el['outputValues'][factor])

        val_new = {}
        for key in vals.keys():
            val_new[key] = self._choose_best(vals[key])

        return val_new
    # -----------------------------------------------------