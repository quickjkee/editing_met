from utils.parser import Parser
from utils.dataset import Dataset

parser = Parser(path_to_datas='files')
df = parser.raw_to_df()
dataset = Dataset(df)
print(dataset)