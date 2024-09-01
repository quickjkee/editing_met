from sklearn.model_selection import train_test_split

from utils.parser import Parser
from utils.dataset import Dataset

from train.src import train

# Prepare dataset
# -------------------------------------------------
root = '/extra_disk_1/quickjkee/projects/consistency_inversion_editing/results'
parser = Parser(path_to_datas=f'{root}/sbs_editing/results')
df = parser.raw_to_df()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_dataset = Dataset(train_df,
                        local_path=root)
print(train_dataset[0])
test_dataset = Dataset(test_df,
                       local_path=root)
# -------------------------------------------------


# Train
train.run_train(train_dataset,
                test_dataset,
                test_dataset)
