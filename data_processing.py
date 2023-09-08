from utils import read_textfile, vis_data,cvat2yolo, create_dir, text2yolo, get_images,text2toml,toml2cvac,combine_folders,toml2cvact,sort_videos,get_images_per_time_of_day,select_training_images,select_anno_train_images,move_images
from scipy.stats import skew
import os



class data_proc():
    def __init__(self,video_folder):
        self.video_folder=video_folder


    def visualize_data(self):
        # Visualize data
        vis_data(video_folder, anno_csv)
    
    def sort_videos(self):
        sort_videos(self.video_folder)
        print('========================')
        print('Video Sorting Completed')
        print('========================')


    def train_frame_selection(self):
        select_training_images()
        print('========================')
        print('Frame Selection Completed')
        print('========================')

    def convert_text2toml(self,anno,width,height):
        text2toml(anno,width,height)
        print('===========================')
        print('TOML conversion Completed')
        print('===========================') 


    def convert_toml2cvat(self):
        cls=['motorbike','DHelmet','DNoHelmet','P1Helmet','P1NoHelmet','P2Helmet','P2NoHelmet']
        toml2cvact('images','selected toml files', 'cvats',cls)
        print('===========================')
        print('CVAT conversion Completed')
        print('===========================')        
    

    def convert_text2yolo(self,anno,width,height):
        text2toml(anno,width,height)
        print('=================================')
        print('Text 2 YOLO conversion Completed')
        print('=================================') 

    def get_train_toml(self):
        select_anno_train_images()
        print('=================================')
        print('Train Toml conversion Completed')
        print('=================================')         

    def move_imgs(self):
        move_images()
        print('=================================')
        print('Image Folder Move Completed')
        print('=================================')

    def convert_cvat_to_yolo(self):
        for i in range(1, 101):
            folder_name = str(i)
            json_file = f'cvats/{folder_name}/demo.json'
            img_path = f'cvats/{folder_name}/images/'
            cvat2yolo(json_file, img_path,i)
            print("done")



if __name__ == "__main__":
    
    width=1920
    height=1080

    video_folder=r'C:\Users\HP\PycharmProjects\pythonProject1\project_iitmandi\Videos'
    anno='gt1_final_f.txt'
    anno_csv=read_textfile('gt1_final_f.txt')


    proc=data_proc(video_folder)
    proc.visualize_data()
    #proc.sort_videos()
    #proc.train_frame_selection()
    #proc.convert_text2toml(anno,width,height)
    #proc.get_train_toml()
    #proc.move_imgs()
    #proc.convert_toml2cvat()
    #proc.convert_cvat_to_yolo()

    """def convert_cvat_to_yolo(self):
        dsts_folder = 'cvats'  # Replace with the actual path to the destination folder
        json_files = os.listdir(dsts_folder)
        img_path = 'images'  # Replace with the actual path to the image folder

        for json_file in json_files:
            if json_file.endswith('.json'):
                json_path = os.path.join(dsts_folder, json_file)
                cvat2yolo(json_path, img_path)
                print(f'Converted {json_file} to YOLO format')"""

    import os
    import toml

    # Mapping of classes to their YOLO indices
    class_mapping = {
        "motorbike": 0,
        "DHelmet": 1,
        "DNoHelmet": 2,
        "P1Helmet": 3,
        "P1NoHelmet": 4,
        "P2Helmet": 5,
        "P2NoHelmet": 6,
    }


    def convert_to_yolo_format(objects):
        yolo_lines = []
        for obj in objects:
            class_name = obj["class"]
            if class_name in class_mapping:
                class_index = class_mapping[class_name]
                xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]

                x_center = (xmin + xmax) / (2.0 * 1920)
                y_center = (ymin + ymax) / (2.0 * 1080)
                x_width = (xmax - xmin) / 1920
                y_height = (ymax - ymin) / 1080

                yolo_lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {x_width:.6f} {y_height:.6f}")

        return yolo_lines


    input_parent_folder = "yolo_anno_1"  # Update this with the actual path
    output_folder = "alldata"  # Update this with the actual path

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    output_count = 0

    for folder_number in range(1, 101):
        folder_name = str(folder_number)
        folder_path = os.path.join(input_parent_folder, folder_name)

        if os.path.exists(folder_path):
            for toml_file in os.listdir(folder_path):
                if toml_file.lower().endswith(".toml"):
                    toml_filename = os.path.join(folder_path, toml_file)

                    with open(toml_filename, "r") as f:
                        toml_data = toml.load(f)  # Load the TOML data from the file

                    # Extract objects from toml_data
                    objects = toml_data.get("objects", [])

                    yolo_lines = convert_to_yolo_format(objects)

                    output_count += 1
                    output_filename = f"{output_count}.txt"
                    output_path = os.path.join(output_folder, output_filename)

                    with open(output_path, "w") as f:
                        f.write("\n".join(yolo_lines))

                    print(f"Converted {toml_filename} and saved YOLO format to {output_path}")
        else:
            print(f"Folder not found for folder {folder_name}")

    print("Conversion completed.")


