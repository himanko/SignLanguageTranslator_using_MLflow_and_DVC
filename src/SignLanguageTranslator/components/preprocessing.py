import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from SignLanguageTranslator.entity.config_entity import PreprocessingConfig
from SignLanguageTranslator.utils.prepcessing_utils import load_relevant_data_subset
from SignLanguageTranslator.utils.common import IDX_MAP
from SignLanguageTranslator import logger

# For now choose 30 as random, but later will take average over entire dataset
FIXED_FRAMES = 30

class Preprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_preprocess = FeaturePreprocess()

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

            # Save the CSV file inside self.config.com_dir
            csv_filename = os.path.join(self.config.com_dir, 'train.csv')

            train_df.to_csv(csv_filename, index=False)

            # Convert Signs to Orginal Encodings
            train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes

            # Dictionaries to translate sign <-> ordinal encoded sign
            SIGN2ORD = train_df[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
            ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

            # # Checking the output of SIGN2ORD and ORD2SIGN
            # print(SIGN2ORD)  # Translate sign name to ordinal encoding
            # print(ORD2SIGN)  # Translate ordinal encoding to sign name

            return train_df, SIGN2ORD, ORD2SIGN

        except Exception as e:
            raise e

    # Load parquet file, preprocess it and return (features, labels)
    def convert_row(self, row):
        # Here row is the tuple from from iterrows()
        x = torch.tensor(load_relevant_data_subset(row[1].path))
        # Convert NaN values to 0
        x[np.isnan(x)] = 0
        x = self.feature_preprocess(x).cpu().numpy()
        return x, row[1].sign_ord
    # Preprocess on entire dataset
    def convert_and_save_data(self, df):
        try:
            # df = self.generate_labels.get_generate_labels()
            total = df.shape[0]
            npdata = np.zeros((total, 30, 104, 3))
            nplabels = np.zeros(total)

            for i, row in tqdm(enumerate(df.iterrows()), total=total):
                #print(row[1].file_path, row[1].sign_ord)
                (x,y) = self.convert_row(row)
                npdata[i,:,:,:] = x
                nplabels[i] = y

            # Save features and labels as npy files
            np.save(os.path.join(self.config.com_dir, "feature_data.npy"), npdata)
            np.save(os.path.join(self.config.com_dir, "feature_labels.npy"), nplabels)

        except Exception as e:
            raise e

    def apply(self):
        try:
            # train_df, SIGN2ORD = self.get_generate_labels()
            train_df = self.get_generate_labels()
            self.convert_and_save_data(train_df)

            logger.info("Feature data and labels saved successfully.")
        except Exception as e:
            logger.error(f"Error occurred during preprocessing: {e}")




        


class FeaturePreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        n_frames = x_in.shape[0]

        # Normalization to a common mean
        x_in = x_in - x_in[~torch.isnan(x_in)].mean(0,keepdim=True)
        x_in = x_in / x_in[~torch.isnan(x_in)].std(0, keepdim=True)

        # Landmarks reduction
        lips     = x_in[:, IDX_MAP['lips']]
        lhand    = x_in[:, IDX_MAP['left_hand']]
        pose     = x_in[:, IDX_MAP['upper_body']]
        rhand    = x_in[:, IDX_MAP['right_hand']]
        x_in = torch.cat([lips,
                          lhand,
                          pose,
                          rhand], 1) # (n_frames, n_landmarks, 3)

        # Replace nan with 0 before Interpolation
        x_in[torch.isnan(x_in)] = 0

        # If n_frames < k, use linear interpolation,
        # else, use nearest neighbor interpolation
        x_in = x_in.permute(2,1,0) #(3, n_landmarks, n_frames)
        if n_frames < FIXED_FRAMES:
            x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode= 'linear')
        else:
            x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode= 'nearest-exact')

        return x_in.permute(2,1,0) # (n_frames, n_landmarks, 3)