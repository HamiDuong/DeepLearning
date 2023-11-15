import pandas as pd
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def xml_to_df(path):
    xml_list = []
    for file in os.listdir(path):
        xml_path = os.path.join(path, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def split_df_into_train_test(df):
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df_test = pd.DataFrame(columns = column_name)
    df_train = pd.DataFrame(columns = column_name)

    for elem in df["class"].value_counts().index.values:
        filtered_df = df[df["class"] == elem]
        train, test = train_test_split(filtered_df, test_size = 0.2, random_state = 42)
        df_train = pd.concat([df_train, train], ignore_index = True)
        df_test = pd.concat([df_test, test], ignore_index = True)
    
    return df_train, df_test

def main():
    xml_path = "Fullscreen_Screenshots/XML"
    df = xml_to_df(xml_path)
    train, test = split_df_into_train_test(df)
    train.to_csv(('Outputs/train.csv'), index=None)
    test.to_csv(('Outputs/test.csv'), index=None)

main()
