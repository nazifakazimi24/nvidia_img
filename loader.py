
from PIL import Image

from os import listdir
from os.path import isdir, join
import pandas as pd

from project import clip_vit_model

def load_image(foldername):
    """
    Load images from the dataset folders.
    :param foldername: path of the main images folder file.
    :return: A DataFrame df that contains the image's query, topic, rank, and relative path.
    """

    df = pd.DataFrame()
    folders = [join(foldername, f) for f in listdir(foldername) if isdir(join(foldername, f))]
    folders.sort()
    folders = [folders[0]]  #uncomment this to continue with all the folders
    for folder in folders:
        folder_list = [join(folder, f) for f in listdir(folder) if isdir(join(folder, f))]
        for fl in folder_list:
            fname = listdir(fl+'/pages')[0]
            if fname[0] != '.':
                filepath = fl + '/pages/' + fname + '/rankings.jsonl'
                imagepath = fl + '/image.webp'
                
                ranking = pd.read_json(path_or_buf=filepath, lines=True)
                ranking['image_id'] = fl.split('/')[-1]
                ranking['image_path'] = imagepath

                df = pd.concat([df, ranking], ignore_index=True)
    return df

def load_topic(path):
    """
    Load topics from topics.xml file.
    :param path: path of the topics.xml file.
    :return: A DataFrame df that contains only the topic's number and title.
    """
    topics = pd.read_xml(path_or_buffer = path)
    return topics[['number', 'title']]

def load_test():
    data_df = load_image("../ImCap/images")
    topics_df = load_topic("../ImCap/topics.xml")
    data_df = data_df.merge(topics_df, left_on='topic', right_on='number')

    # index of the test image, for individual testing
    test_idx = 0

    # add the desired test queries
    test_query = data_df[['query', 'title']].values[test_idx].tolist()
    # test_query.append('infographic')
    # test_query.append('global warming')
    # test_query.append('coral reefs')
    # test_query.append('climate change')
    # test_query.append('man-made climate change')

    # load the test image
    test_image_path = data_df['image'].values[test_idx]
    test_image = Image.open(test_image_path)
    print("test path: ", test_image_path)

    # print the result
    print("___OUTPUT CLIP MODEL:___")
    clip_vit_model(test_image, test_query)

if __name__ == "__main__":
    load_test()
