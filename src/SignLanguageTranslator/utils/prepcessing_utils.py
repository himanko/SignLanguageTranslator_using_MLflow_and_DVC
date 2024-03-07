import os
import pandas as pd
import numpy as np
from SignLanguageTranslator.entity.config_entity import PreprocessingConfig

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    try:
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        # Convert NaN values to 0
        data = data.fillna(0)
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)
    
    except Exception as e:
            raise e
    
class GenerateLabels:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    

    def get_generate_labels(self):
        try:
     
            signs = []
            paths = []

            # root_folder = self.config.root_dir
            root_folder = os.listdir(self.config.root_dir)

            for class_name in root_folder:
                class_path = os.path.join(root_folder, class_name)
                if not os.path.isdir(class_path):
                    continue

                for file_name in os.listdir(class_path):
                    phrase = class_name
                    # path = f"/kaggle/input/landmarks-wlasl/landmarks_files_(alphabatically)/WLASL_300/{class_name}/{file_name}"
                    path = os.path.join(self.config.root_dir, class_name, file_name)
                    signs.append(phrase)
                    paths.append(path)
            data = {
                'sign': signs,
                'path': paths,
            }

            train_df = pd.DataFrame(data)

            csv_filename = 'train.csv'
            train_df.to_csv(csv_filename, index=False)

            # Convert Signs to Orginal Encodings
            train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes

            # Dictionaries to translate sign <-> ordinal encoded sign
            SIGN2ORD = train_df[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
            ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

            # Checking the output of SIGN2ORD and ORD2SIGN
            print(SIGN2ORD)  # Translate sign name to ordinal encoding
            print(ORD2SIGN)  # Translate ordinal encoding to sign name

            

        except Exception as e:
            raise e

