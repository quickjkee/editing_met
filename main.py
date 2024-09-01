from utils.parser import Parser
from utils.dataset import Dataset

root = '/extra_disk_1/quickjkee/projects/consistency_inversion_editing/results'

parser = Parser(path_to_datas=f'{root}/sbs_editing/results')
df = parser.raw_to_df()
dataset = Dataset(df,
                  local_path=root)
print(dataset[0])