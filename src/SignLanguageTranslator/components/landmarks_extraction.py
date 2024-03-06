import os
import cv2
import pandas as pd
from SignLanguageTranslator.entity.config_entity import LandmarksExtractionConfig
from SignLanguageTranslator.components.landmarks_utils import landmarks_data, initialize_mediapipe
from SignLanguageTranslator import logger





class LandmarksExtraction:
    def __init__(self, config: LandmarksExtractionConfig):
        self.config = config
        self.mp_drawing, self.mp_drawing_styles,self.mp_holistic, self.mp_hands, self.holistic, self.hands = initialize_mediapipe()
        self.video_df_list = []

    
    def process_video(self, video_file, class_name, file_name_label):
        try:
            comp_folder = self.config.com_dir
            cap = cv2.VideoCapture(video_file)
            video_frames = []
            frame_no = 0

            while cap.isOpened():
                print('\r', frame_no, file_name_label, end='')
                success, image = cap.read()

                if not success:
                    break

                image = cv2.resize(image, dsize=None, fx=4, fy=4)
                height, width, _ = image.shape

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Use MediaPipe Holistic to get face, hands, and pose landmarks
                # result_holistic = self.holistic.process(image)

                # Use MediaPipe Hands to get left and right hand landmarks
                result_hands = self.hands.process(image)

                data = []
                fy = height / width

                

                for landmark_type, landmarks, num_landmarks in [
                                                            ('left_hand', result_hands.multi_hand_landmarks[0] if result_hands.multi_hand_landmarks else None, 21),
                                                            ('right_hand', result_hands.multi_hand_landmarks[1] if (result_hands.multi_hand_landmarks and len(result_hands.multi_hand_landmarks) > 1) else None, 21)]:
                    if landmarks is None:
                        for i in range(num_landmarks):
                            data.append(landmarks_data(landmark_type, i, 0, 0, 0, file_name_label))
                    else:
                        assert len(landmarks.landmark) == num_landmarks
                        for i in range(num_landmarks):
                            xyz = landmarks.landmark[i]
                            x = xyz.x if hasattr(xyz, 'x') else 0
                            y = xyz.y * fy if hasattr(xyz, 'y') else 0
                            z = xyz.z if hasattr(xyz, 'z') else 0
                            data.append(landmarks_data(landmark_type, i, x, y, z, file_name_label))

                # Convert landmarks_data objects to dictionaries
                data_dicts = []
                for data_object in data:
                    data_dicts.append(data_object.__dict__)


                frame_df = pd.DataFrame(data_dicts)
                frame_df.loc[:, 'frame'] = frame_no
                video_frames.append(frame_df)

                frame_no += 1

            video_df = pd.concat(video_frames, ignore_index=True)
            self.video_df_list.append(video_df)

            output_folder = os.path.join(comp_folder, class_name.split('.')[-1].strip())
            os.makedirs(output_folder, exist_ok=True)
            parquet_file_path = os.path.join(output_folder, f"{file_name_label}.parquet")
            os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)
            video_df.to_parquet(parquet_file_path)

            cap.release()

            logger.info(f"Processed video: {video_file}")

        except Exception as e:
            raise e
        

    def apply(self):
        try:
            comp_folder = self.config.com_dir
            root_folder = self.config.root_dir

            for class_name in os.listdir(root_folder):
                class_name_path = os.path.join(root_folder, class_name)
                if not os.path.isdir(class_name_path):
                    continue

                for file_name in os.listdir(class_name_path):
                    file_extension = file_name.split('.')[-1]

                    if file_extension == 'extra':
                        extra_path = os.path.join(class_name_path, 'extra')
                        for files in os.listdir(extra_path):
                            file_name_label = files.split('.')[0]
                            parquet_file = os.path.join(comp_folder, class_name.split('.')[-1].strip(), file_name_label + '.parquet')

                            if not os.path.exists(parquet_file):
                                video_file = os.path.join(extra_path, files)
                                print(f"Processing video: {video_file}")
                                self.process_video(video_file, class_name, file_name_label)
                            else:
                                print(f"Skipping existing parquet file: {parquet_file}")

                    elif file_extension in {'MOV', 'AVI', 'mp4'}:
                        file_name_label = file_name.split('.')[0]
                        parquet_file = os.path.join(comp_folder, class_name.split('.')[-1].strip(), file_name_label + '.parquet')

                        if not os.path.exists(parquet_file):
                            video_file = os.path.join(class_name_path, file_name)
                            print(f"Processing video: {video_file}")
                            self.process_video(video_file, class_name, file_name_label)
                        else:
                            print(f"Skipping existing parquet file: {parquet_file}")

            # Print to check the length of video_df_list
            print(f"Number of DataFrames in video_df_list: {len(self.video_df_list)}")

            self.holistic.close()

            final_df = pd.concat(self.video_df_list, ignore_index=True)
            if not self.video_df_list:
                print("No data to save.")
            else:
                final_df.to_csv('output.csv', index=False)

        except Exception as e:
            raise e


        
        
